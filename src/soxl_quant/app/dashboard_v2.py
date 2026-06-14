from __future__ import annotations

import json
import os
import pickle
import platform
import shutil
import subprocess
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
MODEL_RETENTION_GATE_SUMMARY = (
    "runtime/model_selection/card/model_retention_walkforward_gate_v1/summary.json"
)
MODEL_RETENTION_FRONTIER_SUMMARY = (
    "runtime/model_selection/card/model_retention_frontier_v1/summary.json"
)
DASHBOARD_TASK_LABELS = {
    "preopen": "com.soxl.preopen.scheduler",
    "postclose": "com.soxl.daily.scheduler",
    "dashboard_refresh": "com.soxl.dashboard.refresh",
}
MODEL_DATE_FIELDS = ("datetime", "date")
REQUIRED_MODEL_FIELDS = ("strategy_nav", "buyhold_nav", "position", "action", "price")
MODEL_NON_FEATURE_FIELDS = {
    *MODEL_DATE_FIELDS,
    *REQUIRED_MODEL_FIELDS,
    "window",
    "excess",
    "excess_pct",
    "feature_snapshot",
    "open",
    "high",
    "low",
    "close",
    "volume",
}
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
VERSIONED_DASHBOARD_CONTEXT_FIELDS = {
    *REFERENCE_FEATURE_COLUMNS,
    *FEATURE_COLS,
    "rsi6",
    "ma180",
    "hv_20",
    "vol_coef",
    "panic_score",
    "market_state",
}
RETAINED_DASHBOARD_EXPORT_SOURCE_KIND = "retained_dashboard_export"
RETAINED_DASHBOARD_EXPORT_REASON = "evidence_only_retained_dashboard_export"
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


def _retained_dashboard_model_sources(root: Path) -> dict[str, dict[str, Any]]:
    versioned_dir = root / "dashboard_data" / "versioned"
    if not versioned_dir.exists():
        return {}
    sources: dict[str, dict[str, Any]] = {}
    for detail_path in sorted(versioned_dir.glob("*/SOXL.json")):
        profile_name = detail_path.parent.name
        if profile_name in {SOXL_BACKTEST_REFERENCE_PROFILE, "paper_trading"}:
            continue
        try:
            payload = json.loads(detail_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta = payload.get("meta") or payload.get("_meta") or {}
        if isinstance(meta, dict) and meta.get("source_kind") == "post_train_review":
            continue
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            continue
        sources[profile_name] = {
            "profile": profile_name,
            "status": "retained_dashboard_export",
            "output": _relative_to(root, detail_path),
            "result": "current",
            "source_kind": RETAINED_DASHBOARD_EXPORT_SOURCE_KIND,
            "display_name": profile_name,
            "runnable": False,
            "runnable_reason": RETAINED_DASHBOARD_EXPORT_REASON,
            "format": "dashboard_versioned_json",
        }
    return sources


def _load_model_retention_gate(root: Path) -> dict[str, Any] | None:
    path = root / MODEL_RETENTION_GATE_SUMMARY
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    by_profile = {
        str(row.get("profile")): row
        for row in rows
        if isinstance(row, dict) and row.get("profile")
    }
    if not isinstance(summary, dict):
        summary = {}
    return {
        "source_path": MODEL_RETENTION_GATE_SUMMARY,
        "summary": summary,
        "by_profile": by_profile,
    }


def _model_retention_gate_manifest(gate: dict[str, Any] | None) -> dict[str, Any] | None:
    if gate is None:
        return None
    summary = gate["summary"]
    return {
        "source_path": gate["source_path"],
        "profiles_scanned": summary.get("profiles_scanned"),
        "passing_profile_count": summary.get("passing_profile_count"),
        "passing_profiles": summary.get("passing_profiles", []),
        "reference_profile": summary.get("reference_profile"),
        "required_best_return_pct": summary.get("required_best_return_pct"),
        "max_walkforward_drawdown_pct": summary.get("max_walkforward_drawdown_pct"),
        "required_wf_windows": summary.get("required_wf_windows", []),
        "registry_keep_actor_count": summary.get("registry_keep_actor_count"),
        "registry_keep_actor_strict_violation_count": summary.get("registry_keep_actor_strict_violation_count"),
        "registry_keep_actor_strict_violations": summary.get("registry_keep_actor_strict_violations", []),
    }


def _load_model_retention_frontier(root: Path) -> dict[str, Any] | None:
    path = root / MODEL_RETENTION_FRONTIER_SUMMARY
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    if not isinstance(summary, dict):
        return None
    return {
        "source_path": MODEL_RETENTION_FRONTIER_SUMMARY,
        "rows_scanned": summary.get("rows_scanned"),
        "completed_handoff_count": summary.get("completed_handoff_count"),
        "recommended_next_source": summary.get("recommended_next_source"),
        "recommended_next_action": summary.get("recommended_next_action"),
        "recommended_primary_blocker": summary.get("recommended_primary_blocker"),
        "required_best_return_pct": summary.get("required_best_return_pct"),
        "reference_return_pct": summary.get("reference_return_pct"),
        "max_walkforward_drawdown_pct": summary.get("max_walkforward_drawdown_pct"),
        "required_wf_windows": summary.get("required_wf_windows", []),
    }


def _model_retention_gate_metadata(profile_name: str, gate: dict[str, Any] | None) -> dict[str, Any]:
    if gate is None:
        return {}
    row = gate["by_profile"].get(profile_name)
    if not isinstance(row, dict):
        return {
            "retention_gate_pass": False,
            "retention_decision": "missing_retention_gate_row",
            "retention_failures": ["missing_retention_gate_row"],
        }
    failures = row.get("failures", [])
    if isinstance(failures, str):
        failures = [item for item in failures.split(";") if item]
    elif not isinstance(failures, list):
        failures = []
    return {
        "retention_gate_pass": bool(row.get("gate_pass", False)),
        "retention_decision": str(row.get("retention_decision") or ""),
        "retention_failures": [str(item) for item in failures],
        "worst_wf_max_drawdown_pct": row.get("worst_wf_max_drawdown_pct"),
        "required_best_return_pct": row.get("required_best_return_pct"),
        "required_soxlbacktest_return_pct": row.get("required_soxlbacktest_return_pct"),
    }


def _schema_signature(columns: list[str]) -> str:
    payload = "|".join(columns).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _feature_metadata(root: Path, profile_name: str, source: dict[str, Any]) -> dict[str, Any]:
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
            "feature_schema_source": "reference_schema",
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
    source_kind = str(source.get("source_kind") or "qlib_model")
    if source_kind == RETAINED_DASHBOARD_EXPORT_SOURCE_KIND:
        runnable = False
        runnable_reason = str(source.get("runnable_reason") or RETAINED_DASHBOARD_EXPORT_REASON)
    else:
        actor = str(source.get("actor") or "").strip()
        actor_exists = bool(actor) and (root / actor).exists()
        runnable = actor_exists
        runnable_reason = "ok" if actor_exists else "missing_actor_artifact" if actor else "missing_actor"
    metadata: dict[str, Any] = {
        "source_kind": source_kind,
        "display_name": source.get("display_name", profile_name),
        "runnable": runnable,
        "runnable_reason": runnable_reason,
        "feature_packs": packs,
        "feature_columns": columns,
        "feature_schema_signature": _schema_signature(columns),
        "feature_schema_source": "profile_yaml" if columns else "missing_profile_schema",
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


def _feature_snapshot_from_source_row(
    source: pd.Series,
    *,
    extra_non_feature_fields: set[str] | frozenset[str] | None = None,
) -> FeatureSnapshot:
    nested = source.get("feature_snapshot")
    if isinstance(nested, dict):
        snapshot = {
            str(key): _json_feature_value(value)
            for key, value in nested.items()
            if _json_feature_value(value) is not None
        }
        if snapshot:
            return snapshot

    snapshot: FeatureSnapshot = {}
    non_feature_fields = MODEL_NON_FEATURE_FIELDS | set(extra_non_feature_fields or ())
    for key, value in source.items():
        column = str(key)
        if column in non_feature_fields:
            continue
        json_value = _json_feature_value(value)
        if json_value is not None:
            snapshot[column] = json_value
    return snapshot


def _feature_columns_from_model_rows(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        snapshot = row.get("feature_snapshot")
        if not isinstance(snapshot, dict):
            continue
        for column, value in snapshot.items():
            if value is not None and column not in columns:
                columns.append(str(column))
    return columns


def _feature_rows_from_model_rows(rows: list[dict[str, Any]]) -> FeatureRowsByDate:
    feature_rows: FeatureRowsByDate = {}
    for row in rows:
        snapshot = row.get("feature_snapshot")
        if not isinstance(snapshot, dict) or not snapshot:
            continue
        date = str(row.get("date") or "")
        if date:
            feature_rows[date] = {str(key): value for key, value in snapshot.items() if value is not None}
    return feature_rows


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


def _feature_summary_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, np.generic):
        value = value.item()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def build_feature_action_summary(
    rows: list[dict[str, Any]],
    *,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    ordered_columns = _dedupe_preserve_order([str(column) for column in feature_columns])
    stats: dict[str, dict[str, list[float]]] = {
        column: {"BUY": [], "SELL": []}
        for column in ordered_columns
    }
    for row in rows:
        action = str(row.get("action") or "").upper()
        if action not in {"BUY", "SELL"}:
            continue
        snapshot = row.get("feature_snapshot")
        if not isinstance(snapshot, dict):
            continue
        for column in ordered_columns:
            value = _feature_summary_number(snapshot.get(column))
            if value is not None:
                stats[column][action].append(value)

    summary: list[dict[str, Any]] = []
    for column in ordered_columns:
        buy_values = stats[column]["BUY"]
        sell_values = stats[column]["SELL"]
        if not buy_values and not sell_values:
            continue
        buy_mean = float(np.mean(buy_values)) if buy_values else None
        sell_mean = float(np.mean(sell_values)) if sell_values else None
        buy_minus_sell = buy_mean - sell_mean if buy_mean is not None and sell_mean is not None else None
        summary.append(
            {
                "feature": column,
                "buy_count": len(buy_values),
                "sell_count": len(sell_values),
                "buy_mean": buy_mean,
                "sell_mean": sell_mean,
                "buy_minus_sell": buy_minus_sell,
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            -abs(float(row["buy_minus_sell"])) if row.get("buy_minus_sell") is not None else 0.0,
            str(row["feature"]),
        ),
    )


def _launchagent_loaded_labels(*, detect_launchd: bool) -> set[str] | None:
    if not detect_launchd:
        return None
    output = os.environ.get("SOXL_DASHBOARD_V2_LAUNCHCTL_LIST")
    if output is None:
        if platform.system() != "Darwin" or shutil.which("launchctl") is None:
            return None
        try:
            completed = subprocess.run(
                ["launchctl", "list"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        output = completed.stdout
    labels = set(DASHBOARD_TASK_LABELS.values())
    return {label for label in labels if label in output}


def _launchagents_dir() -> Path:
    override = os.environ.get("SOXL_DASHBOARD_V2_LAUNCHAGENTS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "LaunchAgents"


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
    if feature_metadata.get("source_kind") not in {"qlib_model", RETAINED_DASHBOARD_EXPORT_SOURCE_KIND}:
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
    rows_by_date = {str(row["date"]): row for row in rows}
    for event in events:
        date = str(event["date"])
        row_snapshot_source = rows_by_date.get(date, {}).get("feature_snapshot")
        if isinstance(row_snapshot_source, dict):
            event_snapshot = _feature_snapshot(row_snapshot_source, snapshot_columns)
            if event_snapshot:
                event["feature_snapshot"] = event_snapshot
                continue
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
        row_snapshot = row.get("feature_snapshot") if isinstance(row.get("feature_snapshot"), dict) else None
        model_snapshot = _feature_snapshot(row_snapshot, snapshot_columns)
        if not model_snapshot:
            model_snapshot = _feature_snapshot((model_feature_rows_by_date or {}).get(date), snapshot_columns)
        snapshot = model_snapshot or _feature_snapshot(market_rows_by_date.get(date), snapshot_columns)
        enriched.append({**row, "feature_snapshot": snapshot})
    return enriched


def _auto_update_status(
    root: Path,
    freshness: dict[str, Any],
    freshness_path: Path,
    *,
    detect_launchd: bool,
) -> dict[str, Any]:
    cron_path = root / "production" / "crontab.txt"
    preopen_plist = root / "production" / "com.soxl.preopen.scheduler.plist"
    postclose_plist = root / "production" / "com.soxl.daily.scheduler.plist"
    dashboard_refresh_plist = root / "production" / "com.soxl.dashboard.refresh.plist"
    scheduler = root / "production" / "run_soxl_scheduler.sh"
    dashboard_refresh = root / "production" / "run_dashboard_refresh.sh"
    dashboard_refresh_installer = root / "production" / "install_dashboard_refresh_launchagent.sh"
    cron_text = cron_path.read_text(encoding="utf-8") if cron_path.exists() else ""
    loaded_labels = _launchagent_loaded_labels(detect_launchd=detect_launchd)
    launchagents_dir = _launchagents_dir()
    launchd_detected = loaded_labels is not None

    def _task(
        *,
        name: str,
        configured: bool,
        schedule: str,
        source_path: Path,
        log_path: str,
        command_path: Path,
    ) -> dict[str, Any]:
        label = DASHBOARD_TASK_LABELS[name]
        installed = (launchagents_dir / f"{label}.plist").exists()
        loaded = label in loaded_labels if loaded_labels is not None else None
        enabled = configured if loaded is None else bool(configured and loaded)
        return {
            "name": name,
            "label": label,
            "configured": bool(configured),
            "installed": bool(installed),
            "loaded": loaded,
            "enabled": enabled,
            "schedule": schedule,
            "source_path": _relative_to(root, source_path),
            "log_path": log_path,
            "command_path": _relative_to(root, command_path),
        }

    tasks = [
        _task(
            name="preopen",
            configured=preopen_plist.exists() or "schedule-slot preopen" in cron_text,
            schedule="21:00 Mon-Fri Asia/Shanghai",
            source_path=preopen_plist if preopen_plist.exists() else cron_path,
            log_path="runtime/soxl/paper_trading/scheduler_preopen.log",
            command_path=scheduler,
        ),
        _task(
            name="postclose",
            configured=postclose_plist.exists() or "schedule-slot postclose" in cron_text,
            schedule="06:15 Tue-Sat Asia/Shanghai",
            source_path=postclose_plist if postclose_plist.exists() else cron_path,
            log_path="runtime/soxl/paper_trading/scheduler_postclose.log",
            command_path=scheduler,
        ),
        _task(
            name="dashboard_refresh",
            configured=dashboard_refresh.exists()
            and (dashboard_refresh_plist.exists() or "dashboard_refresh.log" in cron_text),
            schedule="06:30 Tue-Sat Asia/Shanghai",
            source_path=dashboard_refresh_plist if dashboard_refresh_plist.exists() else cron_path,
            log_path="runtime/soxl/paper_trading/dashboard_refresh.log",
            command_path=dashboard_refresh,
        ),
    ]
    counts = freshness.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}
    profile_rows = freshness.get("profiles", [])
    if not isinstance(profile_rows, list):
        profile_rows = []
    if not counts and profile_rows:
        derived_counts = {"current": 0, "refreshed": 0, "failed": 0, "skipped": 0}
        for profile_row in profile_rows:
            if not isinstance(profile_row, dict):
                continue
            result = str(profile_row.get("result") or "")
            if result in derived_counts:
                derived_counts[result] += 1
        counts = derived_counts
    ready_count = int(counts.get("current", 0) or 0) + int(counts.get("refreshed", 0) or 0)
    return {
        "configured": any(bool(task["configured"]) for task in tasks),
        "timezone": "Asia/Shanghai",
        "launchd_detected": launchd_detected,
        "launchagents_dir": str(launchagents_dir),
        "tasks": tasks,
        "manual_refresh_command": _relative_to(root, dashboard_refresh),
        "install_command": _relative_to(root, dashboard_refresh_installer),
        "run_now_command": f"{_relative_to(root, dashboard_refresh_installer)} --run-now",
        "refresh_pipeline": [
            "scripts/refresh_dashboard_model_replays.py",
            "scripts/build_dashboard_bundle.py",
            "scripts/check_dashboard_static_export.py",
        ],
        "model_replay_freshness": {
            "generated_at": freshness.get("generated_at"),
            "latest_data_date": freshness.get("latest_data_date"),
            "safe_backtest_end": freshness.get("safe_backtest_end"),
            "counts": counts,
            "profile_count": len(profile_rows),
            "ready_count": ready_count,
            "source_path": _relative_to(root, freshness_path),
        },
        "status_paths": {
            "production_goal_gates": _relative_to(root, root / "dashboard_data" / "production_goal_gates.json"),
            "production_status": _relative_to(root, root / "dashboard_data" / "production_status.json"),
        },
        "remote_deploy_required": False,
    }


def _unavailable_profiles_from_freshness(
    root: Path,
    freshness: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = freshness.get("profiles", [])
    if not isinstance(rows, list):
        return []
    unavailable: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile") or "").strip()
        if not profile:
            continue
        result = str(row.get("result") or "").strip() or "unknown"
        output = str(row.get("output") or "").strip()
        output_exists = bool(output) and (root / output).exists()
        if result in {"current", "refreshed"} and output_exists:
            continue
        reason = str(row.get("reason") or "").strip()
        if result in {"current", "refreshed"} and not output_exists:
            reason = reason or "missing_output_artifact"
        item = {
            "profile": profile,
            "status": str(row.get("status") or ""),
            "result": result,
            "reason": reason or "not_replayable",
        }
        for field in ("actor", "profile_yaml", "output"):
            value = str(row.get(field) or "").strip()
            if value:
                item[field] = value
        unavailable.append(item)
    return unavailable


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
    return _normalize_model_frame(
        frame,
        path_label=str(path),
        safe_backtest_end=safe_backtest_end,
        window=window,
    )


def normalize_dashboard_version_rows(
    path: Path,
    *,
    safe_backtest_end: str | None,
    window: str | None = None,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing dashboard version JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(raw_rows, list):
        raise ValueError(f"dashboard version JSON must contain rows list: {path}")
    return _normalize_model_frame(
        pd.DataFrame(raw_rows),
        path_label=str(path),
        safe_backtest_end=safe_backtest_end,
        window=window,
        extra_non_feature_fields=VERSIONED_DASHBOARD_CONTEXT_FIELDS,
    )


def _normalize_model_frame(
    frame: pd.DataFrame,
    *,
    path_label: str,
    safe_backtest_end: str | None,
    window: str | None = None,
    extra_non_feature_fields: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
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
        source_feature_snapshot = _feature_snapshot_from_source_row(
            source,
            extra_non_feature_fields=extra_non_feature_fields,
        )
        if first_strategy_nav is None:
            first_strategy_nav = strategy_nav
        excess = strategy_nav - buyhold_nav
        excess_pct = (strategy_nav / first_strategy_nav - 1.0) if first_strategy_nav else 0.0
        row = {
            "date": source[date_field].strftime("%Y-%m-%d"),
            "strategy_nav": strategy_nav,
            "buyhold_nav": buyhold_nav,
            "position": position,
            "action": action,
            "price": price,
            "excess": excess,
            "excess_pct": excess_pct,
        }
        if source_feature_snapshot:
            row["feature_snapshot"] = source_feature_snapshot
        rows.append(row)
    if not rows:
        raise ValueError(f"no replay rows remain after safe_backtest_end filter: {path_label}")
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


def _profile_groups(
    model_payloads: dict[str, dict[str, Any]],
    profiles: list[str],
) -> dict[str, list[str]]:
    groups = {"runnable": [], "reference": [], "evidence_only": []}
    for profile in profiles:
        payload = model_payloads.get(profile, {})
        if payload.get("runnable"):
            groups["runnable"].append(profile)
            continue
        source_kind = str(payload.get("source_kind") or "")
        runnable_reason = str(payload.get("runnable_reason") or "")
        if source_kind in {"soxlbacktest_qlib_buysell", "reference_baseline"} or runnable_reason == "reference_baseline":
            groups["reference"].append(profile)
        else:
            groups["evidence_only"].append(profile)
    return groups


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
    detect_launchd: bool = False,
) -> DashboardV2Result:
    root = root.resolve()
    output_dir = output_dir.resolve()
    freshness_path = freshness_path or root / "dashboard_data" / "model_replay_freshness.json"
    freshness = load_replay_freshness(freshness_path)
    retention_gate = _load_model_retention_gate(root)
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
    for profile_name, source in _retained_dashboard_model_sources(root).items():
        by_name.setdefault(profile_name, source)
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
        if source.get("format") == "dashboard_versioned_json":
            rows = normalize_dashboard_version_rows(
                replay_path,
                safe_backtest_end=str(safe_backtest_end) if safe_backtest_end else None,
                window=source.get("window"),
            )
        else:
            rows = normalize_model_rows(
                replay_path,
                safe_backtest_end=str(safe_backtest_end) if safe_backtest_end else None,
                window=source.get("window"),
            )
        feature_metadata = _feature_metadata(root, profile_name, source)
        model_feature_rows = model_feature_rows_by_profile.get(profile_name)
        feature_snapshot_error = None
        inline_feature_rows = _feature_rows_from_model_rows(rows)
        if inline_feature_rows:
            inline_columns = _feature_columns_from_model_rows(rows)
            model_feature_rows = inline_feature_rows
            if profile_name == SOXL_BACKTEST_REFERENCE_PROFILE and inline_columns:
                feature_metadata = {
                    **feature_metadata,
                    "feature_columns": inline_columns,
                    "feature_schema_signature": _schema_signature(inline_columns),
                    "feature_schema_source": "replay_inline_snapshot",
                }
            else:
                feature_metadata = {
                    **feature_metadata,
                    "feature_schema_source": "replay_inline_snapshot"
                    if not feature_metadata.get("feature_columns")
                    else f"{feature_metadata.get('feature_schema_source')};replay_inline_snapshot",
                }
        if model_feature_rows is None:
            model_feature_rows, feature_snapshot_error = _load_model_feature_rows(
                profile_name,
                feature_metadata,
                root / "data" / "qlib" / "soxl",
            )
            if model_feature_rows:
                feature_metadata = {
                    **feature_metadata,
                    "feature_schema_source": f"{feature_metadata.get('feature_schema_source')};qlib_feature_frame",
                }
        inferred_columns = _model_feature_columns(model_feature_rows)
        if not feature_metadata.get("feature_columns") and inferred_columns:
            inferred_source = str(feature_metadata.get("feature_schema_source") or "")
            feature_metadata = {
                **feature_metadata,
                "feature_columns": inferred_columns,
                "feature_schema_signature": _schema_signature(inferred_columns),
                "feature_schema_source": "replay_inline_snapshot"
                if "replay_inline_snapshot" in inferred_source
                else "inferred_snapshot",
            }
        snapshot_columns = _snapshot_columns_for_model(
            [str(column) for column in feature_metadata.get("feature_columns", [])],
            market_payload_rows,
            model_feature_rows,
        )
        if not snapshot_columns and inferred_columns:
            snapshot_columns = inferred_columns
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
            **_model_retention_gate_metadata(profile_name, retention_gate),
            "snapshot_columns": snapshot_columns,
            "source_path": _relative_to(root, replay_path),
            "replay_start": rows[0]["date"],
            "replay_end": rows[-1]["date"],
            "row_count": len(rows),
            "rows": enriched_rows,
            "feature_action_summary": build_feature_action_summary(
                enriched_rows,
                feature_columns=snapshot_columns,
            ),
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
    retention_frontier = _load_model_retention_frontier(root)
    default_chart_profile = next(
        (
            profile_name
            for profile_name in defaults
            if bool(model_payloads.get(profile_name, {}).get("runnable"))
        ),
        None,
    ) or next(
        (
            profile_name
            for profile_name in defaults
            if model_payloads.get(profile_name, {}).get("source_kind") != "soxlbacktest_qlib_buysell"
        ),
        defaults[0],
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latest_market_date": latest_market_date,
        "safe_backtest_end": safe_backtest_end,
        "available_profiles": profiles_to_write,
        "unavailable_profiles": _unavailable_profiles_from_freshness(root, freshness),
        "profile_groups": _profile_groups(model_payloads, profiles_to_write),
        "default_baseline_profile": baseline,
        "default_chart_profile": default_chart_profile,
        "default_comparison_profiles": defaults,
        "model_retention_gate": _model_retention_gate_manifest(retention_gate),
        "model_retention_frontier": retention_frontier,
        "auto_update": _auto_update_status(
            root,
            freshness,
            freshness_path,
            detect_launchd=detect_launchd,
        ),
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
