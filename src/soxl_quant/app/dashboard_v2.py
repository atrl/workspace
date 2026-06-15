from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.build_backtest_dashboard import FEATURE_COLS


SOXL_BACKTEST_REFERENCE_PROFILE = "soxl_backtest_policy_reimplementation_v1"
SOXL_BACKTEST_REFERENCE_WINDOW = "ytd_2026"
SOXL_BACKTEST_REFERENCE_DAILY = (
    "runtime/production/card/soxl_backtest_policy_reimplementation_v1/new_policy_daily.csv"
)
DIAGNOSTIC_PROFILE_PREFERENCES = (
    "balanced_rule_residual_v22",
    "balanced_rule_residual_v19_reg",
    "balanced_rule_residual_v19",
)
MAX_DEFAULT_COMPARISON_PROFILES = 4
MODEL_DATE_FIELDS = ("datetime", "date")
REQUIRED_MODEL_FIELDS = ("strategy_nav", "buyhold_nav", "position", "action", "price")
ACTION_VALUES = {"BUY", "SELL", "HOLD"}
SELL_ACTION_ALIASES = {"TAKE_PROFIT", "STOP_LOSS", "STOP", "TRIM", "REDUCE"}
FINAL_RETENTION_POLICY = "final_model_retained_or_reference"
STRICT_RETENTION_GATE_PATH = (
    "runtime/model_selection/card/model_retention_walkforward_gate_v1/summary.json"
)


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
        }
    }


def _summary_candidates(root: Path, profile_name: str, source: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    summary_path = source.get("summary_path")
    if summary_path:
        candidates.append(root / str(summary_path))
    output = source.get("output")
    if output:
        card_dir = (root / str(output)).parent
        candidates.extend(
            [
                card_dir / f"soxl_{profile_name}_summary.json",
                card_dir / f"{profile_name}_summary.json",
                card_dir / "summary.json",
            ]
        )
        candidates.extend(sorted(card_dir.glob("*_summary.json")))
    deduped: list[Path] = []
    for path in candidates:
        if path not in deduped:
            deduped.append(path)
    return deduped


def _final_retention_reason(decision: dict[str, Any]) -> str:
    reason = decision.get("reason") or decision.get("block_reason")
    if reason:
        return str(reason)
    reasons = decision.get("reasons")
    if isinstance(reasons, list) and reasons:
        return "; ".join(str(item) for item in reasons)
    status = decision.get("status")
    return str(status) if status else "final model retention gate blocked"


def _profile_retention_status(root: Path, profile_name: str, source: dict[str, Any]) -> dict[str, Any]:
    if profile_name == SOXL_BACKTEST_REFERENCE_PROFILE or source.get("status") == "reference":
        return {
            "allowed": True,
            "code": "reference_baseline",
            "status": "reference_baseline",
            "reason": "SOXLBacktest reference buy/sell baseline",
            "summary_path": None,
            "final_model_retained": True,
        }

    summary_path = next((path for path in _summary_candidates(root, profile_name, source) if path.exists()), None)
    if summary_path is None:
        return {
            "allowed": False,
            "code": "missing_summary",
            "status": "blocked",
            "reason": "missing summary artifact with final_model_retention_decision",
            "summary_path": None,
            "final_model_retained": False,
        }

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "allowed": False,
            "code": "invalid_summary_json",
            "status": "blocked",
            "reason": f"invalid summary JSON: {exc.msg}",
            "summary_path": _relative_to(root, summary_path),
            "final_model_retained": False,
        }
    decision = summary.get("final_model_retention_decision") if isinstance(summary, dict) else None
    if not isinstance(decision, dict):
        return {
            "allowed": False,
            "code": "missing_final_retention_decision",
            "status": "blocked",
            "reason": "summary lacks final_model_retention_decision",
            "summary_path": _relative_to(root, summary_path),
            "final_model_retained": False,
        }

    final_model_retained = decision.get("final_model_retained") is True
    status = str(decision.get("status") or ("passed" if final_model_retained else "blocked"))
    return {
        "allowed": final_model_retained,
        "code": "final_model_retained" if final_model_retained else "final_retention_blocked",
        "status": status,
        "reason": _final_retention_reason(decision),
        "summary_path": _relative_to(root, summary_path),
        "final_model_retained": final_model_retained,
    }


def _public_retention_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": status["code"],
        "status": status["status"],
        "reason": status["reason"],
        "summary_path": status.get("summary_path"),
        "final_model_retained": status["final_model_retained"],
    }


def _load_strict_retention_gate(root: Path) -> dict[str, Any] | None:
    path = root / STRICT_RETENTION_GATE_PATH
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(summary, dict):
        return None
    passing_count = int(summary.get("passing_profile_count") or 0)
    profiles_scanned = int(summary.get("profiles_scanned") or 0)
    return {
        "status": "passed" if passing_count > 0 else "blocked",
        "source_path": STRICT_RETENTION_GATE_PATH,
        "profiles_scanned": profiles_scanned,
        "passing_profile_count": passing_count,
        "passing_profiles": [str(profile) for profile in summary.get("passing_profiles") or []],
        "reference_profile": summary.get("reference_profile"),
        "reference_return_pct": summary.get("reference_return_pct"),
        "current_best_return_pct": summary.get("current_best_return_pct"),
        "best_return_ratio": summary.get("best_return_ratio"),
        "required_best_return_pct": summary.get("required_best_return_pct"),
        "max_walkforward_drawdown_pct": summary.get("max_walkforward_drawdown_pct"),
        "required_wf_windows": [str(window) for window in summary.get("required_wf_windows") or []],
        "closest_wf_drawdown_candidate": summary.get("closest_wf_drawdown_candidate"),
        "closest_ytd_return_candidate": summary.get("closest_ytd_return_candidate"),
    }


def _diagnostic_defaults(
    by_name: dict[str, dict[str, Any]],
    retention_by_name: dict[str, dict[str, Any]],
    selected_profiles: list[str] | None,
) -> list[str]:
    if selected_profiles:
        candidates = [profile for profile in selected_profiles if profile in by_name]
    else:
        candidates = list(DIAGNOSTIC_PROFILE_PREFERENCES)
    return [
        profile
        for profile in _dedupe_preserve_order(candidates)
        if profile in by_name and not retention_by_name[profile]["allowed"]
    ]


def _model_safe_backtest_end(
    source: dict[str, Any],
    retention_status: dict[str, Any],
    safe_backtest_end: str | None,
) -> str | None:
    if source.get("status") == "reference" or retention_status["code"] == "reference_baseline":
        return None
    return str(safe_backtest_end) if safe_backtest_end else None


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


def _raw_symbol_frame(root: Path, etf: str) -> pd.DataFrame | None:
    csv_path = root / "data" / "raw" / "symbols" / f"{etf.upper()}.csv"
    if not csv_path.exists():
        return None
    frame = pd.read_csv(csv_path, parse_dates=["date"])
    required = ["date", "open", "high", "low", "close", "volume"]
    if not all(column in frame.columns for column in required):
        return None
    return frame[required].sort_values("date").reset_index(drop=True)


def _overlay_raw_symbol_prices(frame: pd.DataFrame, raw: pd.DataFrame | None) -> pd.DataFrame:
    if raw is None or raw.empty:
        return frame
    base = frame.copy()
    base["date"] = pd.to_datetime(base["date"])
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    base = base.set_index("date").sort_index()
    raw = raw.set_index("date").sort_index()
    for date in raw.index:
        if date not in base.index:
            base.loc[date, :] = np.nan
    for column in ["open", "high", "low", "close", "volume"]:
        base.loc[raw.index, column] = raw[column]
    return base.sort_index().reset_index()


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
        frame = _overlay_raw_symbol_prices(frame, _raw_symbol_frame(root, etf))
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
        frame["vix"] = frame["vix"].ffill().bfill().fillna(20.0)
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
        "max_drawdown": max_drawdown,
        "avg_position": position_sum / len(rows),
        "row_count": float(len(rows)),
    }


def _rebase_nav(value: float, start_value: float) -> float:
    if start_value == 0:
        return value
    return value / start_value * 100.0


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
    rows: list[dict[str, Any]] = []
    running_drawdowns: dict[str, float] = {profile: 0.0 for profile in profiles}
    running_peaks: dict[str, float] = {
        profile: float(by_profile[profile][common_dates[0]]["strategy_nav"])
        for profile in profiles
    }
    start_navs: dict[str, float] = {
        profile: float(by_profile[profile][common_dates[0]]["strategy_nav"])
        for profile in profiles
    }
    for date in common_dates:
        baseline_row = baseline_rows[date]
        baseline_nav = float(baseline_row["strategy_nav"])
        baseline_nav_rebased = _rebase_nav(baseline_nav, start_navs[baseline_profile])
        running_peaks[baseline_profile] = max(running_peaks[baseline_profile], baseline_nav)
        running_drawdowns[baseline_profile] = baseline_nav / running_peaks[baseline_profile] - 1.0
        action_mismatch_count = 0
        profile_values: dict[str, dict[str, Any]] = {}
        for profile in profiles:
            row = by_profile[profile][date]
            nav = float(row["strategy_nav"])
            nav_rebased = _rebase_nav(nav, start_navs[profile])
            running_peaks[profile] = max(running_peaks[profile], nav)
            running_drawdowns[profile] = nav / running_peaks[profile] - 1.0
            action_mismatch = row["action"] != baseline_row["action"]
            action_mismatch_count += int(action_mismatch)
            profile_values[profile] = {
                "strategy_nav": nav,
                "strategy_nav_rebased": nav_rebased,
                "position": row["position"],
                "action": row["action"],
                "nav_delta_vs_baseline": nav_rebased - baseline_nav_rebased,
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
    selected_profiles: list[str] | None = None,
    baseline_profile: str | None = None,
) -> DashboardV2Result:
    root = root.resolve()
    output_dir = output_dir.resolve()
    freshness_path = freshness_path or root / "dashboard_data" / "model_replay_freshness.json"
    freshness = load_replay_freshness(freshness_path)
    safe_backtest_end = freshness.get("safe_backtest_end")
    canonical_data_date = freshness.get("latest_data_date")
    market_payload_rows = normalize_market_rows(
        market_rows or load_ohlc_and_indicators_for_root(root, "SOXL"),
        feature_rows if feature_rows is not None else load_feature_rows(root),
    )
    market_latest_date = market_payload_rows[-1]["date"]
    latest_market_date = max(
        str(canonical_data_date) if canonical_data_date else market_latest_date,
        market_latest_date,
    )
    ready_profiles = [
        profile
        for profile in freshness["profiles"]
        if profile.get("result") in {"current", "refreshed"}
    ]
    by_name = {str(profile["profile"]): profile for profile in ready_profiles}
    by_name.update(_reference_model_sources(root))
    retention_by_name = {
        profile_name: _profile_retention_status(root, profile_name, source)
        for profile_name, source in by_name.items()
    }
    allowed_by_name = {
        profile_name: source
        for profile_name, source in by_name.items()
        if retention_by_name[profile_name]["allowed"]
    }
    if selected_profiles:
        defaults = _dedupe_preserve_order(selected_profiles)
    else:
        preferred = [
            SOXL_BACKTEST_REFERENCE_PROFILE,
            "balanced_rule_residual_v22",
            "smoke",
        ]
        defaults = [name for name in preferred if name in allowed_by_name][:2]
        if len(defaults) < 2:
            defaults.extend(name for name in allowed_by_name if name not in defaults)
    if baseline_profile and baseline_profile in allowed_by_name and baseline_profile not in defaults:
        defaults = [baseline_profile] + defaults
    deployable_defaults = [profile for profile in _dedupe_preserve_order(defaults) if profile in allowed_by_name]
    diagnostic_defaults = _diagnostic_defaults(by_name, retention_by_name, selected_profiles)
    defaults = _dedupe_preserve_order([*deployable_defaults, *diagnostic_defaults])[:MAX_DEFAULT_COMPARISON_PROFILES]
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
        baseline = defaults[0]
    models_dir = output_dir / "models"
    if models_dir.exists():
        selected_model_files = {f"{profile_name}.json" for profile_name in defaults}
        for stale_path in models_dir.glob("*.json"):
            if stale_path.name not in selected_model_files:
                stale_path.unlink()
    model_payloads: dict[str, dict[str, Any]] = {}
    model_paths: dict[str, Path] = {}
    for profile_name in defaults:
        source = by_name[profile_name]
        retention_status = retention_by_name[profile_name]
        replay_path = root / str(source["output"])
        rows = normalize_model_rows(
            replay_path,
            safe_backtest_end=_model_safe_backtest_end(source, retention_status, safe_backtest_end),
            window=source.get("window"),
        )
        payload = {
            "profile": profile_name,
            "status": source.get("status"),
            "actor": source.get("actor"),
            "profile_yaml": source.get("profile_yaml"),
            "source_path": _relative_to(root, replay_path),
            "replay_start": rows[0]["date"],
            "replay_end": rows[-1]["date"],
            "row_count": len(rows),
            "rows": rows,
            "events": derive_model_events(rows),
            "metrics": {"full": compute_metrics(rows)},
            "retention": _public_retention_status(retention_status),
        }
        if source.get("window"):
            payload["window"] = source.get("window")
        model_payloads[profile_name] = payload
        model_path = output_dir / "models" / f"{profile_name}.json"
        _write_json(model_path, payload)
        model_paths[profile_name] = model_path
    comparison = build_comparison_payload(
        model_payloads,
        baseline_profile=baseline,
        compared_profiles=defaults,
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latest_market_date": latest_market_date,
        "canonical_data_date": canonical_data_date,
        "safe_backtest_end": safe_backtest_end,
        "available_profiles": defaults,
        "deployable_profiles": [profile for profile in defaults if retention_by_name[profile]["allowed"]],
        "diagnostic_profiles": [profile for profile in defaults if not retention_by_name[profile]["allowed"]],
        "filtered_profile_count": max(0, len(by_name) - len(defaults)),
        "retention_filter": {
            "policy": FINAL_RETENTION_POLICY,
            "input_profile_count": len(by_name),
            "allowed_profile_count": len(allowed_by_name),
            "blocked_profile_count": len(by_name) - len(allowed_by_name),
        },
        "blocked_profiles": {
            profile_name: _public_retention_status(status)
            for profile_name, status in sorted(retention_by_name.items())
            if not status["allowed"]
        },
        "default_baseline_profile": baseline,
        "default_comparison_profiles": defaults,
        "profile_summaries": {
            profile_name: {
                "status": payload.get("status"),
                "retention": payload["retention"],
                "replay_start": payload["replay_start"],
                "replay_end": payload["replay_end"],
                "row_count": payload["row_count"],
                "source_path": payload["source_path"],
                "window": payload.get("window"),
            }
            for profile_name, payload in model_payloads.items()
        },
        "source_paths": {
            "model_replay_freshness": _relative_to(root, freshness_path),
            "market": "data/processed/panels/soxl_soxx_vix_full.csv",
            **{
                f"model:{profile_name}": payload["source_path"]
                for profile_name, payload in model_payloads.items()
            },
        },
    }
    strict_retention_gate = _load_strict_retention_gate(root)
    if strict_retention_gate is not None:
        manifest["strict_retention_gate"] = strict_retention_gate
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
