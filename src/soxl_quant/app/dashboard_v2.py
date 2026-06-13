from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.build_backtest_dashboard import FEATURE_COLS
from src.soxl_quant.config import load_profile
from src.soxl_quant.data import build_dataset, feature_columns_for_packs, init_qlib


SOXL_BACKTEST_REFERENCE_PROFILE = "soxl_backtest_policy_reimplementation_v1"
SOXL_BACKTEST_REFERENCE_WINDOW = "ytd_2026"
SOXL_BACKTEST_REFERENCE_DAILY = (
    "runtime/production/card/soxl_backtest_policy_reimplementation_v1/new_policy_daily.csv"
)
MODEL_DATE_FIELDS = ("datetime", "date")
REQUIRED_MODEL_FIELDS = ("strategy_nav", "buyhold_nav", "position", "action", "price")
ACTION_VALUES = {"BUY", "SELL", "HOLD"}
SELL_ACTION_ALIASES = {"TAKE_PROFIT", "STOP_LOSS", "STOP", "TRIM", "REDUCE"}
REFERENCE_FEATURE_PACKS = ("soxlbacktest_qlib_buysell_reference",)
REFERENCE_FEATURE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "rsi6",
    "ma180",
    "hv_20",
    "vol_coef",
    "panic_score",
    "market_state",
    "vix",
)
DEFAULT_SNAPSHOT_COLUMNS = (
    "close",
    "rsi6",
    "rsi",
    "vix",
    "hv_20",
    "volatility",
    "spread",
    "vix_zscore",
    "panic_score",
    "vol_coef",
    "market_state",
    "intraday_range",
)
FEATURE_SNAPSHOT_ALIASES = {
    "return": ("close",),
    "rsi_6": ("rsi6", "rsi"),
    "rsi_14": ("rsi", "rsi6"),
    "realized_vol_10": ("volatility", "hv_20"),
    "realized_vol_21": ("volatility", "hv_20"),
    "realized_vol_63": ("volatility", "hv_20"),
    "vol_coef_21": ("vol_coef",),
    "vix_level": ("vix",),
    "vix_zscore": ("vix_zscore",),
    "intraday_range": ("intraday_range",),
    "mom_21": ("mom_20",),
    "RB_PANIC_SCORE_PROXY": ("panic_score",),
    "RB_VOL_REGIME": ("vol_coef",),
}
FeatureSnapshot = dict[str, str | int | float | bool | None]
FeatureRowsByDate = dict[str, FeatureSnapshot]
FeatureRowsByProfile = dict[str, FeatureRowsByDate]


@dataclass(frozen=True)
class DashboardV2Result:
    output_dir: Path
    manifest_path: Path
    market_path: Path
    model_paths: dict[str, Path]
    comparison_path: Path


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _relative_to(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_float(value: object, field: str, row_index: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric field {field!r} at row {row_index}") from exc
    if pd.isna(number):
        raise ValueError(f"invalid numeric field {field!r} at row {row_index}")
    return number


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _normalize_action(value: object) -> str:
    action = str(value or "HOLD").upper()
    if action in ACTION_VALUES:
        return action
    if action in SELL_ACTION_ALIASES:
        return "SELL"
    return "HOLD"


def _reference_model_sources(root: Path) -> dict[str, dict[str, Any]]:
    daily_path = root / SOXL_BACKTEST_REFERENCE_DAILY
    if not daily_path.exists():
        return {}
    return {
        SOXL_BACKTEST_REFERENCE_PROFILE: {
            "profile": SOXL_BACKTEST_REFERENCE_PROFILE,
            "status": "reference",
            "output": SOXL_BACKTEST_REFERENCE_DAILY,
            "result": "current",
            "window": SOXL_BACKTEST_REFERENCE_WINDOW,
            "source_kind": "soxlbacktest_qlib_buysell",
            "display_name": "SOXLBacktest Buy/Sell Baseline",
            "runnable": False,
            "runnable_reason": "reference_baseline",
        }
    }


def _schema_signature(columns: list[str]) -> str:
    payload = "|".join(columns).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _feature_metadata(profile_name: str, source: dict[str, Any]) -> dict[str, Any]:
    if profile_name == SOXL_BACKTEST_REFERENCE_PROFILE:
        columns = list(REFERENCE_FEATURE_COLUMNS)
        return {
            "source_kind": source.get("source_kind", "soxlbacktest_qlib_buysell"),
            "display_name": source.get("display_name", "SOXLBacktest Buy/Sell Baseline"),
            "runnable": False,
            "runnable_reason": source.get("runnable_reason", "reference_baseline"),
            "feature_packs": list(REFERENCE_FEATURE_PACKS),
            "feature_columns": columns,
            "feature_schema_signature": _schema_signature(columns),
        }

    try:
        profile = load_profile(profile_name)
        packs = list(profile.feature.packs)
        columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        packs = []
        columns = []
        feature_error = str(exc)
    else:
        feature_error = None
    runnable = bool(source.get("actor"))
    metadata: dict[str, Any] = {
        "source_kind": source.get("source_kind", "qlib_model"),
        "display_name": source.get("display_name", profile_name),
        "runnable": runnable,
        "runnable_reason": "ok" if runnable else "missing_actor",
        "feature_packs": packs,
        "feature_columns": columns,
        "feature_schema_signature": _schema_signature(columns),
    }
    if feature_error:
        metadata["feature_error"] = feature_error
    return metadata


def _model_feature_columns(model_feature_rows_by_date: FeatureRowsByDate | None) -> list[str]:
    columns: list[str] = []
    for row in (model_feature_rows_by_date or {}).values():
        for key, value in row.items():
            if value is not None and key not in columns:
                columns.append(str(key))
    return columns


def _snapshot_columns_for_model(
    feature_columns: list[str],
    market_payload_rows: list[dict[str, Any]],
    model_feature_rows_by_date: FeatureRowsByDate | None = None,
) -> list[str]:
    model_columns = _model_feature_columns(model_feature_rows_by_date)
    if model_columns:
        if feature_columns:
            ordered = [column for column in feature_columns if column in model_columns]
            ordered.extend(column for column in model_columns if column not in ordered)
            return ordered
        return model_columns

    available: set[str] = set()
    for row in market_payload_rows:
        available.update(str(key) for key, value in row.items() if key != "date" and value is not None)
    candidates: list[str] = []
    for column in feature_columns:
        candidates.extend(FEATURE_SNAPSHOT_ALIASES.get(column, (column,)))
    candidates.extend(DEFAULT_SNAPSHOT_COLUMNS)
    return [column for column in _dedupe_preserve_order(candidates) if column in available]


def _feature_snapshot(row: dict[str, Any] | None, columns: list[str]) -> dict[str, Any]:
    if not row:
        return {}
    return {column: row[column] for column in columns if column in row}


def _json_feature_value(value: object) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, str):
        return value
    return str(value)


def _feature_frame_to_rows(frame: pd.DataFrame, columns: list[str]) -> FeatureRowsByDate:
    if isinstance(frame.index, pd.MultiIndex):
        if "instrument" in frame.index.names:
            frame = frame.droplevel("instrument")
        else:
            frame = frame.reset_index(level=list(range(1, frame.index.nlevels)), drop=True)
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index)
    available_columns = [column for column in columns if column in frame.columns]
    rows: FeatureRowsByDate = {}
    if not available_columns:
        return rows
    for dt, source in frame[available_columns].sort_index().iterrows():
        snapshot: FeatureSnapshot = {}
        for column in available_columns:
            value = _json_feature_value(source[column])
            if value is not None:
                snapshot[column] = value
        if snapshot:
            rows[pd.Timestamp(dt).strftime("%Y-%m-%d")] = snapshot
    return rows


def _load_model_feature_rows(
    profile_name: str,
    feature_metadata: dict[str, Any],
    provider_uri: Path,
) -> tuple[FeatureRowsByDate, str | None]:
    if feature_metadata.get("source_kind") != "qlib_model":
        return {}, None
    columns = feature_metadata.get("feature_columns")
    packs = feature_metadata.get("feature_packs")
    if not isinstance(columns, list) or not columns:
        return {}, "missing_feature_columns"
    if not isinstance(packs, list):
        packs = []
    if not provider_uri.exists():
        return {}, f"missing_qlib_store: {provider_uri}"
    try:
        init_qlib(provider_uri)
        dataset = build_dataset(feature_packs=tuple(str(pack) for pack in packs), profile_name=profile_name)
        frame = dataset.handler.feature_frame
    except Exception as exc:
        return {}, str(exc)
    return _feature_frame_to_rows(frame, [str(column) for column in columns]), None


def _derive_model_events_with_features(
    rows: list[dict[str, Any]],
    *,
    market_rows_by_date: dict[str, dict[str, Any]],
    snapshot_columns: list[str],
    model_feature_rows_by_date: FeatureRowsByDate | None = None,
) -> list[dict[str, Any]]:
    events = derive_model_events(rows)
    for event in events:
        date = str(event["date"])
        model_snapshot = _feature_snapshot((model_feature_rows_by_date or {}).get(date), snapshot_columns)
        event["feature_snapshot"] = model_snapshot or _feature_snapshot(market_rows_by_date.get(date), snapshot_columns)
    return events


def _attach_row_feature_snapshots(
    rows: list[dict[str, Any]],
    *,
    market_rows_by_date: dict[str, dict[str, Any]],
    snapshot_columns: list[str],
    model_feature_rows_by_date: FeatureRowsByDate | None = None,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        date = str(row["date"])
        model_snapshot = _feature_snapshot((model_feature_rows_by_date or {}).get(date), snapshot_columns)
        snapshot = model_snapshot or _feature_snapshot(market_rows_by_date.get(date), snapshot_columns)
        enriched.append({**row, "feature_snapshot": snapshot})
    return enriched


def _auto_update_status(root: Path, freshness: dict[str, Any], freshness_path: Path) -> dict[str, Any]:
    cron_path = root / "production" / "crontab.txt"
    preopen_plist = root / "production" / "com.soxl.preopen.scheduler.plist"
    postclose_plist = root / "production" / "com.soxl.daily.scheduler.plist"
    dashboard_refresh_plist = root / "production" / "com.soxl.dashboard.refresh.plist"
    scheduler = root / "production" / "run_soxl_scheduler.sh"
    dashboard_refresh = root / "production" / "run_dashboard_refresh.sh"
    cron_text = cron_path.read_text(encoding="utf-8") if cron_path.exists() else ""
    tasks = [
        {
            "name": "preopen",
            "enabled": preopen_plist.exists() or "schedule-slot preopen" in cron_text,
            "schedule": "21:00 Mon-Fri Asia/Shanghai",
            "source_path": _relative_to(root, preopen_plist if preopen_plist.exists() else cron_path),
            "log_path": "runtime/soxl/paper_trading/scheduler_preopen.log",
            "command_path": _relative_to(root, scheduler),
        },
        {
            "name": "postclose",
            "enabled": postclose_plist.exists() or "schedule-slot postclose" in cron_text,
            "schedule": "06:15 Tue-Sat Asia/Shanghai",
            "source_path": _relative_to(root, postclose_plist if postclose_plist.exists() else cron_path),
            "log_path": "runtime/soxl/paper_trading/scheduler_postclose.log",
            "command_path": _relative_to(root, scheduler),
        },
        {
            "name": "dashboard_refresh",
            "enabled": dashboard_refresh.exists()
            and (dashboard_refresh_plist.exists() or "dashboard_refresh.log" in cron_text),
            "schedule": "06:30 Tue-Sat Asia/Shanghai",
            "source_path": _relative_to(root, dashboard_refresh_plist if dashboard_refresh_plist.exists() else cron_path),
            "log_path": "runtime/soxl/paper_trading/dashboard_refresh.log",
            "command_path": _relative_to(root, dashboard_refresh),
        },
    ]
    return {
        "configured": any(bool(task["enabled"]) for task in tasks),
        "timezone": "Asia/Shanghai",
        "tasks": tasks,
        "model_replay_freshness": {
            "generated_at": freshness.get("generated_at"),
            "latest_data_date": freshness.get("latest_data_date"),
            "safe_backtest_end": freshness.get("safe_backtest_end"),
            "counts": freshness.get("counts", {}),
            "source_path": _relative_to(root, freshness_path),
        },
        "status_paths": {
            "production_goal_gates": _relative_to(root, root / "dashboard_data" / "production_goal_gates.json"),
            "production_status": _relative_to(root, root / "dashboard_data" / "production_status.json"),
        },
        "remote_deploy_required": False,
    }


def load_feature_rows(root: Path) -> dict[str, dict[str, float | None]] | None:
    features_path = root / "data" / "features_v2.pkl"
    if not features_path.exists():
        return None
    with features_path.open("rb") as handle:
        frame = pickle.load(handle)
    frame.index = pd.to_datetime(frame.index)
    available = [column for column in FEATURE_COLS if column in frame.columns]
    records: dict[str, dict[str, float | None]] = {}
    for dt, row in frame.iterrows():
        records[dt.strftime("%Y-%m-%d")] = {
            column: float(row[column]) if pd.notna(row[column]) else None
            for column in available
        }
    return records


def load_ohlc_and_indicators_for_root(root: Path, etf: str = "SOXL") -> dict[str, dict[str, Any]]:
    processed_path = root / "data" / "processed" / "panels" / "soxl_soxx_vix_full.csv"
    if etf.upper() == "SOXL" and processed_path.exists():
        frame = pd.read_csv(processed_path, parse_dates=["date"])
        frame = frame.sort_values("date").reset_index(drop=True)
        frame = frame.rename(
            columns={
                "soxl_open": "open",
                "soxl_high": "high",
                "soxl_low": "low",
                "soxl_close": "close",
                "soxl_volume": "volume",
            }
        )
    else:
        csv_path = root / "data" / "raw" / "symbols" / f"{etf.upper()}.csv"
        if not csv_path.exists():
            return {}
        frame = pd.read_csv(csv_path, parse_dates=["date"])
        frame = frame.sort_values("date").reset_index(drop=True)

    delta = frame["close"].diff()
    gain = delta.where(delta > 0, 0).ewm(span=6, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=6, adjust=False).mean()
    frame["rsi6"] = 100 - (100 / (1 + gain / (loss + 1e-5)))
    frame["ma180"] = frame["close"].rolling(window=180).mean()

    returns = np.log(frame["close"] / frame["close"].shift(1))
    frame["hv_20"] = returns.rolling(window=20).std() * np.sqrt(252)
    hv_values = frame["hv_20"].dropna()
    hv_median = float(hv_values.median()) if not hv_values.empty else None
    if hv_median and hv_median > 0:
        frame["vol_coef"] = (frame["hv_20"] / hv_median).clip(0.5, 2.0)
    else:
        frame["vol_coef"] = 1.0

    if "vix" in frame.columns:
        frame["vix"] = frame["vix"].fillna(20.0)
    else:
        frame["vix"] = 20.0

    vix_low, vix_high, fear_min, fear_max, steepness = 10, 30, -100, 100, 0.5
    vix_clipped = frame["vix"].clip(vix_low, vix_high)
    vix_mid = (vix_high + vix_low) / 2
    sigmoid = 1 / (1 + np.exp(-(vix_mid - vix_clipped) * steepness))
    frame["base_panic"] = sigmoid * (fear_max - fear_min) + fear_min

    def _adjust_panic(row: pd.Series) -> float:
        rsi6 = row["rsi6"]
        base = row["base_panic"]
        if pd.isna(rsi6):
            return float(base)
        if rsi6 <= 30:
            return float(max(-150, base - (20 + (30 - rsi6) * 2.5)))
        if rsi6 >= 70:
            return float(min(150, base + (20 + (rsi6 - 70))))
        return float(base)

    frame["panic_score"] = frame.apply(_adjust_panic, axis=1)
    frame["market_state"] = frame["close"] > frame["ma180"]

    result: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = row["date"].strftime("%Y-%m-%d")
        result[key] = {
            "open": float(row["open"]) if pd.notna(row["open"]) else None,
            "high": float(row["high"]) if pd.notna(row["high"]) else None,
            "low": float(row["low"]) if pd.notna(row["low"]) else None,
            "close": float(row["close"]) if pd.notna(row["close"]) else None,
            "volume": int(row["volume"]) if pd.notna(row["volume"]) else None,
            "rsi6": float(row["rsi6"]) if pd.notna(row["rsi6"]) else None,
            "ma180": float(row["ma180"]) if pd.notna(row["ma180"]) else None,
            "hv_20": float(row["hv_20"]) if pd.notna(row["hv_20"]) else None,
            "vol_coef": float(row["vol_coef"]) if pd.notna(row["vol_coef"]) else None,
            "panic_score": float(row["panic_score"]) if pd.notna(row["panic_score"]) else None,
            "market_state": bool(row["market_state"]) if pd.notna(row["market_state"]) else None,
            "vix": float(row["vix"]) if pd.notna(row["vix"]) else None,
        }
    return result


def load_replay_freshness(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing replay freshness file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("profiles"), list):
        raise ValueError("model_replay_freshness.json must contain profiles list")
    return payload


def normalize_market_rows(
    market_rows: dict[str, dict[str, Any]],
    feature_rows: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not market_rows:
        raise ValueError("missing market OHLCV data")
    feature_rows = feature_rows or {}
    rows: list[dict[str, Any]] = []
    for date in sorted(market_rows):
        source = dict(market_rows[date])
        row: dict[str, Any] = {"date": date}
        for field in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi6",
            "ma180",
            "hv_20",
            "vol_coef",
            "panic_score",
            "market_state",
            "vix",
        ):
            row[field] = source.get(field)
        for field in FEATURE_COLS:
            if field in feature_rows.get(date, {}):
                row[field] = feature_rows[date][field]
        rows.append(row)
    if len({row["date"] for row in rows}) != len(rows):
        raise ValueError("duplicate market dates")
    return rows


def normalize_model_rows(
    path: Path,
    *,
    safe_backtest_end: str | None,
    window: str | None = None,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing replay CSV: {path}")
    frame = pd.read_csv(path)
    missing = [field for field in REQUIRED_MODEL_FIELDS if field not in frame.columns]
    if missing:
        raise ValueError(f"missing replay CSV columns: {', '.join(missing)}")
    date_field = next((field for field in MODEL_DATE_FIELDS if field in frame.columns), None)
    if date_field is None:
        raise ValueError(f"missing replay CSV date column: {' or '.join(MODEL_DATE_FIELDS)}")
    if window is not None and "window" in frame.columns:
        frame = frame[frame["window"].astype(str).eq(window)].copy()
    frame[date_field] = pd.to_datetime(frame[date_field], errors="raise")
    frame = frame.sort_values(date_field)
    if frame[date_field].duplicated().any():
        raise ValueError(f"duplicate model dates in {path}")
    if safe_backtest_end is not None:
        frame = frame[frame[date_field] <= pd.Timestamp(safe_backtest_end)]
    rows: list[dict[str, Any]] = []
    first_strategy_nav: float | None = None
    for idx, source in frame.reset_index(drop=True).iterrows():
        strategy_nav = _as_float(source["strategy_nav"], "strategy_nav", idx)
        buyhold_nav = _as_float(source["buyhold_nav"], "buyhold_nav", idx)
        position = _as_float(source["position"], "position", idx)
        price = _as_float(source["price"], "price", idx)
        action = _normalize_action(source.get("action"))
        if first_strategy_nav is None:
            first_strategy_nav = strategy_nav
        excess = strategy_nav - buyhold_nav
        excess_pct = (strategy_nav / first_strategy_nav - 1.0) if first_strategy_nav else 0.0
        rows.append(
            {
                "date": source[date_field].strftime("%Y-%m-%d"),
                "strategy_nav": strategy_nav,
                "buyhold_nav": buyhold_nav,
                "position": position,
                "action": action,
                "price": price,
                "excess": excess,
                "excess_pct": excess_pct,
            }
        )
    if not rows:
        raise ValueError(f"no replay rows remain after safe_backtest_end filter: {path}")
    return rows


def derive_model_events(
    rows: list[dict[str, Any]],
    *,
    material_position_delta: float = 0.10,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_position = 0.0
    for row in rows:
        position = float(row["position"])
        delta = position - previous_position
        action = str(row.get("action") or "HOLD").upper()
        if action in {"BUY", "SELL"}:
            events.append(
                {
                    "date": row["date"],
                    "kind": action,
                    "price": row.get("price"),
                    "position": position,
                    "position_delta": delta,
                }
            )
        if abs(delta) >= material_position_delta:
            events.append(
                {
                    "date": row["date"],
                    "kind": "POSITION_INCREASE" if delta > 0 else "POSITION_DECREASE",
                    "price": row.get("price"),
                    "position": position,
                    "position_delta": delta,
                }
            )
        previous_position = position
    return events


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    start_nav = float(rows[0]["strategy_nav"])
    end_nav = float(rows[-1]["strategy_nav"])
    start_buyhold_nav = float(rows[0].get("buyhold_nav", start_nav))
    end_buyhold_nav = float(rows[-1].get("buyhold_nav", end_nav))
    peak = start_nav
    max_drawdown = 0.0
    position_sum = 0.0
    for row in rows:
        nav = float(row["strategy_nav"])
        peak = max(peak, nav)
        if peak:
            max_drawdown = min(max_drawdown, nav / peak - 1.0)
        position_sum += float(row["position"])
    return {
        "return": end_nav / start_nav - 1.0 if start_nav else 0.0,
        "buyhold_return": end_buyhold_nav / start_buyhold_nav - 1.0 if start_buyhold_nav else 0.0,
        "excess_return": (
            end_nav / start_nav - end_buyhold_nav / start_buyhold_nav
            if start_nav and start_buyhold_nav
            else 0.0
        ),
        "max_drawdown": max_drawdown,
        "avg_position": position_sum / len(rows),
        "row_count": float(len(rows)),
    }


def build_comparison_payload(
    model_payloads: dict[str, dict[str, Any]],
    *,
    baseline_profile: str,
    compared_profiles: list[str],
) -> dict[str, Any]:
    if baseline_profile not in model_payloads:
        raise ValueError(f"baseline profile not available: {baseline_profile}")
    profiles = [profile for profile in _dedupe_preserve_order(compared_profiles) if profile in model_payloads]
    date_sets = [{row["date"] for row in model_payloads[profile]["rows"]} for profile in profiles]
    common_dates = sorted(set.intersection(*date_sets)) if date_sets else []
    if not common_dates:
        raise ValueError("no overlapping date range between selected model rows")
    by_profile = {
        profile: {row["date"]: row for row in model_payloads[profile]["rows"]}
        for profile in profiles
    }
    baseline_rows = by_profile[baseline_profile]
    start_strategy_nav = {
        profile: float(by_profile[profile][common_dates[0]]["strategy_nav"])
        for profile in profiles
    }
    start_buyhold_nav = {
        profile: float(by_profile[profile][common_dates[0]].get("buyhold_nav", start_strategy_nav[profile]))
        for profile in profiles
    }
    rows: list[dict[str, Any]] = []
    running_drawdowns: dict[str, float] = {profile: 0.0 for profile in profiles}
    running_peaks: dict[str, float] = {
        profile: float(by_profile[profile][common_dates[0]]["strategy_nav"])
        for profile in profiles
    }
    for date in common_dates:
        baseline_row = baseline_rows[date]
        baseline_nav = float(baseline_row["strategy_nav"])
        baseline_buyhold_nav = float(baseline_row.get("buyhold_nav", baseline_nav))
        baseline_return = baseline_nav / start_strategy_nav[baseline_profile] - 1.0
        running_peaks[baseline_profile] = max(running_peaks[baseline_profile], baseline_nav)
        running_drawdowns[baseline_profile] = baseline_nav / running_peaks[baseline_profile] - 1.0
        action_mismatch_count = 0
        profile_values: dict[str, dict[str, Any]] = {}
        for profile in profiles:
            row = by_profile[profile][date]
            nav = float(row["strategy_nav"])
            buyhold_nav = float(row.get("buyhold_nav", nav))
            strategy_return = nav / start_strategy_nav[profile] - 1.0
            buyhold_return = buyhold_nav / start_buyhold_nav[profile] - 1.0
            running_peaks[profile] = max(running_peaks[profile], nav)
            running_drawdowns[profile] = nav / running_peaks[profile] - 1.0
            action_mismatch = row["action"] != baseline_row["action"]
            action_mismatch_count += int(action_mismatch)
            profile_values[profile] = {
                "strategy_nav": nav,
                "buyhold_nav": buyhold_nav,
                "position": row["position"],
                "action": row["action"],
                "nav_delta_vs_baseline": nav - baseline_nav,
                "buyhold_nav_delta_vs_baseline": buyhold_nav - baseline_buyhold_nav,
                "strategy_return": strategy_return,
                "buyhold_return": buyhold_return,
                "return_delta_vs_baseline": strategy_return - baseline_return,
                "return_delta_vs_buyhold": strategy_return - buyhold_return,
                "position_delta_vs_baseline": row["position"] - baseline_row["position"],
                "action_mismatch": action_mismatch,
                "drawdown_delta_vs_baseline": running_drawdowns[profile] - running_drawdowns[baseline_profile],
            }
        rows.append({"date": date, "profiles": profile_values, "action_mismatch_count": action_mismatch_count})
    rankings = sorted(
        (
            {"profile": profile, **compute_metrics([by_profile[profile][date] for date in common_dates])}
            for profile in profiles
        ),
        key=lambda item: item["return"],
        reverse=True,
    )
    return {
        "baseline_profile": baseline_profile,
        "compared_profiles": profiles,
        "common_start": common_dates[0],
        "common_end": common_dates[-1],
        "rows": rows,
        "rankings": rankings,
    }


def build_dashboard_v2_data(
    *,
    root: Path,
    output_dir: Path,
    freshness_path: Path | None = None,
    market_rows: dict[str, dict[str, Any]] | None = None,
    feature_rows: dict[str, dict[str, Any]] | None = None,
    model_feature_rows_by_profile: FeatureRowsByProfile | None = None,
    selected_profiles: list[str] | None = None,
    baseline_profile: str | None = None,
) -> DashboardV2Result:
    root = root.resolve()
    output_dir = output_dir.resolve()
    freshness_path = freshness_path or root / "dashboard_data" / "model_replay_freshness.json"
    freshness = load_replay_freshness(freshness_path)
    safe_backtest_end = freshness.get("safe_backtest_end")
    latest_market_date = freshness.get("latest_data_date")
    market_payload_rows = normalize_market_rows(
        market_rows or load_ohlc_and_indicators_for_root(root, "SOXL"),
        feature_rows if feature_rows is not None else load_feature_rows(root),
    )
    market_rows_by_date = {row["date"]: row for row in market_payload_rows}
    ready_profiles = [
        profile
        for profile in freshness["profiles"]
        if profile.get("result") in {"current", "refreshed"}
    ]
    by_name = {
        str(profile["profile"]): profile
        for profile in ready_profiles
        if (root / str(profile.get("output", ""))).exists()
    }
    by_name.update(_reference_model_sources(root))
    if selected_profiles:
        defaults = _dedupe_preserve_order(selected_profiles)
    else:
        preferred = [
            SOXL_BACKTEST_REFERENCE_PROFILE,
            "smoke",
            "balanced_rule_residual_v22",
        ]
        defaults = [name for name in preferred if name in by_name]
        if len(defaults) < 2:
            defaults.extend(name for name in by_name if name not in defaults)
    if baseline_profile and baseline_profile not in defaults:
        defaults = [baseline_profile] + defaults
    defaults = [profile for profile in _dedupe_preserve_order(defaults) if profile in by_name][:4]
    if not defaults:
        raise ValueError("no replayable profiles selected for dashboard v2")
    baseline = baseline_profile or (
        SOXL_BACKTEST_REFERENCE_PROFILE
        if SOXL_BACKTEST_REFERENCE_PROFILE in defaults
        else "balanced_rule_residual_v22"
        if "balanced_rule_residual_v22" in defaults
        else defaults[0]
    )
    if baseline not in defaults:
        raise ValueError(f"baseline profile not selected or replayable: {baseline}")
    profiles_to_write = _dedupe_preserve_order(defaults + sorted(name for name in by_name if name not in defaults))
    models_dir = output_dir / "models"
    expected_model_files = {models_dir / f"{profile_name}.json" for profile_name in profiles_to_write}
    if models_dir.exists():
        for stale_path in models_dir.glob("*.json"):
            if stale_path not in expected_model_files:
                stale_path.unlink()
    model_payloads: dict[str, dict[str, Any]] = {}
    model_paths: dict[str, Path] = {}
    model_feature_rows_by_profile = model_feature_rows_by_profile or {}
    for profile_name in profiles_to_write:
        source = by_name[profile_name]
        replay_path = root / str(source["output"])
        rows = normalize_model_rows(
            replay_path,
            safe_backtest_end=str(safe_backtest_end) if safe_backtest_end else None,
            window=source.get("window"),
        )
        feature_metadata = _feature_metadata(profile_name, source)
        model_feature_rows = model_feature_rows_by_profile.get(profile_name)
        feature_snapshot_error = None
        if model_feature_rows is None:
            model_feature_rows, feature_snapshot_error = _load_model_feature_rows(
                profile_name,
                feature_metadata,
                root / "data" / "qlib" / "soxl",
            )
        snapshot_columns = _snapshot_columns_for_model(
            [str(column) for column in feature_metadata.get("feature_columns", [])],
            market_payload_rows,
            model_feature_rows,
        )
        enriched_rows = _attach_row_feature_snapshots(
            rows,
            market_rows_by_date=market_rows_by_date,
            snapshot_columns=snapshot_columns,
            model_feature_rows_by_date=model_feature_rows,
        )
        payload = {
            "profile": profile_name,
            "status": source.get("status"),
            "actor": source.get("actor"),
            "profile_yaml": source.get("profile_yaml"),
            **feature_metadata,
            "snapshot_columns": snapshot_columns,
            "source_path": _relative_to(root, replay_path),
            "replay_start": rows[0]["date"],
            "replay_end": rows[-1]["date"],
            "row_count": len(rows),
            "rows": enriched_rows,
            "events": _derive_model_events_with_features(
                rows,
                market_rows_by_date=market_rows_by_date,
                snapshot_columns=snapshot_columns,
                model_feature_rows_by_date=model_feature_rows,
            ),
            "metrics": {"full": compute_metrics(rows)},
        }
        if feature_snapshot_error:
            payload["feature_snapshot_error"] = feature_snapshot_error
        if source.get("window"):
            payload["window"] = source.get("window")
        model_payloads[profile_name] = payload
        model_path = output_dir / "models" / f"{profile_name}.json"
        _write_json(model_path, payload)
        model_paths[profile_name] = model_path
    comparison = build_comparison_payload(
        model_payloads,
        baseline_profile=baseline,
        compared_profiles=profiles_to_write,
    )
    default_chart_profile = next(
        (
            profile_name
            for profile_name in defaults
            if bool(model_payloads.get(profile_name, {}).get("runnable"))
        ),
        defaults[0],
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latest_market_date": latest_market_date,
        "safe_backtest_end": safe_backtest_end,
        "available_profiles": profiles_to_write,
        "default_baseline_profile": baseline,
        "default_chart_profile": default_chart_profile,
        "default_comparison_profiles": defaults,
        "auto_update": _auto_update_status(root, freshness, freshness_path),
        "source_paths": {
            "model_replay_freshness": _relative_to(root, freshness_path),
            "market": "data/processed/panels/soxl_soxx_vix_full.csv",
            **{
                f"model:{profile_name}": payload["source_path"]
                for profile_name, payload in model_payloads.items()
            },
        },
    }
    market_payload = {"symbol": "SOXL", "rows": market_payload_rows}
    manifest_path = output_dir / "manifest.json"
    market_path = output_dir / "market" / "SOXL.json"
    comparison_path = output_dir / "comparison" / "default.json"
    _write_json(manifest_path, manifest)
    _write_json(market_path, market_payload)
    _write_json(comparison_path, comparison)
    return DashboardV2Result(
        output_dir=output_dir,
        manifest_path=manifest_path,
        market_path=market_path,
        model_paths=model_paths,
        comparison_path=comparison_path,
    )
