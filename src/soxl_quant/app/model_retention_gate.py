from __future__ import annotations

"""Retention gate review for replayable SOXL models.

The gate is intentionally stricter than the historical runtime-protection
registry: a model can remain protected for production reasons while still being
reported as not promotable under the current walk-forward/drawdown objective.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


PRIMARY_REFERENCE_PROFILE = "soxl_backtest_policy_reimplementation_v1"
REQUIRED_WF_WINDOWS = ("wf_2020", "wf_2022", "wf_2023_2024")
REFERENCE_METRICS_PATH = (
    "runtime/production/card/soxl_backtest_policy_reimplementation_v1/legacy_comparison_metrics.csv"
)
COMPARISON_METRICS_PATH = (
    "runtime/production/card/current_model_ytd_june_comparison_v1/all_model_comparison_metrics.csv"
)
FRESHNESS_PATH = "dashboard_data/model_replay_freshness.json"
REGISTRY_PATH = "runtime/harness/card/registry/profile.yaml"
YTD_WINDOW = "ytd_2026_common"
RETURN_COLUMNS = ("model_return_pct", "current_model_return_pct")
MAX_DRAWDOWN_COLUMNS = ("current_model_max_drawdown_pct", "model_max_drawdown_pct")


@dataclass(frozen=True)
class ReferenceBaseline:
    return_pct: float
    max_drawdown_pct: float
    source_path: str


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _threshold_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "_")


def load_reference_baseline(root: Path) -> ReferenceBaseline:
    path = root / REFERENCE_METRICS_PATH
    frame = pd.read_csv(path)
    matches = frame[frame["window"].astype(str).eq("ytd_2026")]
    if matches.empty:
        raise ValueError(f"{path} does not contain ytd_2026")
    row = matches.iloc[0]
    return ReferenceBaseline(
        return_pct=float(row["new_return_pct"]),
        max_drawdown_pct=float(row.get("new_max_drawdown_pct", 0.0)),
        source_path=REFERENCE_METRICS_PATH,
    )


def load_ytd_comparison(root: Path) -> pd.DataFrame:
    path = root / COMPARISON_METRICS_PATH
    frame = pd.read_csv(path)
    rows = frame[frame["window"].astype(str).eq(YTD_WINDOW)].copy()
    if rows.empty:
        raise ValueError(f"{path} does not contain {YTD_WINDOW}")
    return rows


def _first_existing_column(frame_or_row: Any, columns: tuple[str, ...]) -> str | None:
    available = frame_or_row.columns if hasattr(frame_or_row, "columns") else frame_or_row.index
    for column in columns:
        if column in available:
            return column
    return None


def _load_replayable_profiles(root: Path) -> list[dict[str, Any]]:
    payload = _load_json(root / FRESHNESS_PATH)
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError(f"{FRESHNESS_PATH} must contain profiles list")
    return [
        profile
        for profile in profiles
        if isinstance(profile, dict) and profile.get("result") in {"current", "refreshed"}
    ]


def _load_registry_keep_actor_entries(root: Path) -> list[dict[str, Any]]:
    path = root / REGISTRY_PATH
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = payload.get("profiles", [])
    if not isinstance(entries, list):
        raise ValueError(f"{REGISTRY_PATH} must contain profiles list")
    return [
        entry
        for entry in entries
        if isinstance(entry, dict) and bool(entry.get("keep_actor", False)) and entry.get("name")
    ]


def _wf_report_path(root: Path, profile: str) -> Path | None:
    card_dir = root / "runtime" / "model" / "card" / profile
    candidates = [
        card_dir / f"{profile}_report.json",
        card_dir / f"{profile}-report.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _metric_from_summary(summary: dict[str, Any], window: str) -> dict[str, float] | None:
    if "main_backtest" in summary:
        main = summary.get("main_backtest") or {}
        buyhold = summary.get("buy_and_hold") or {}
        return {
            "return_pct": float(main.get("return", 0.0)),
            "max_drawdown_pct": float(main.get("max_dd", 0.0)),
            "buyhold_return_pct": float(buyhold.get("return", 0.0)),
            "relative_return_pct": float(summary.get("relative_outperformance", 0.0)),
        }
    valid = summary.get("valid") or summary.get("test_proxy")
    if isinstance(valid, dict):
        return {
            "return_pct": float(valid.get("return", 0.0)),
            "max_drawdown_pct": float(valid.get("max_dd", 0.0)),
            "buyhold_return_pct": float(valid.get("buy_and_hold_return", 0.0)),
            "relative_return_pct": float(valid.get("relative_return", 0.0)),
        }
    if window in summary and isinstance(summary[window], dict):
        return _metric_from_summary(summary[window], window)
    return None


def load_walkforward_metrics(root: Path, profile: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    report_path = _wf_report_path(root, profile)
    if report_path is not None:
        report = _load_json(report_path)
        for window in REQUIRED_WF_WINDOWS:
            if window in report and isinstance(report[window], dict):
                extracted = _metric_from_summary(report[window], window)
                if extracted is not None:
                    metrics[window] = extracted

    card_dir = root / "runtime" / "model" / "card" / profile
    for window in REQUIRED_WF_WINDOWS:
        if window in metrics:
            continue
        for path in sorted(card_dir.glob(f"*{window}*summary.json")):
            extracted = _metric_from_summary(_load_json(path), window)
            if extracted is not None:
                metrics[window] = extracted
                break
    return metrics


def _reference_row(reference: ReferenceBaseline) -> dict[str, Any]:
    return {
        "profile": PRIMARY_REFERENCE_PROFILE,
        "status": "reference",
        "source_path": reference.source_path,
        "ytd_return_pct": round(reference.return_pct, 6),
        "ytd_max_drawdown_pct": round(reference.max_drawdown_pct, 6),
        "current_best_return_pct": None,
        "required_best_return_pct": None,
        "required_soxlbacktest_return_pct": round(reference.return_pct, 6),
        "wf_windows_present": "",
        "worst_wf_max_drawdown_pct": None,
        "gate_pass": True,
        "retention_decision": "keep_reference_baseline",
        "failures": [],
    }


def _registry_keep_actor_audit(rows: Iterable[dict[str, Any]], registry_entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_profile = {str(row["profile"]): row for row in rows}
    audit: list[dict[str, Any]] = []
    for entry in registry_entries:
        profile = str(entry["name"])
        gate_row = rows_by_profile.get(profile)
        if gate_row is None:
            audit.append(
                {
                    "profile": profile,
                    "registry_status": str(entry.get("status", "")),
                    "keep_actor": True,
                    "strict_retention_allowed": False,
                    "strict_retention_decision": "block_missing_replay_evidence",
                    "gate_pass": False,
                    "ytd_return_pct": None,
                    "worst_wf_max_drawdown_pct": None,
                    "wf_windows_present": "",
                    "failures": ["missing_replay_evidence"],
                    "notes_ref": str(entry.get("notes_ref", "")),
                }
            )
            continue

        failures = gate_row.get("failures") if isinstance(gate_row.get("failures"), list) else []
        gate_pass = bool(gate_row.get("gate_pass"))
        audit.append(
            {
                "profile": profile,
                "registry_status": str(entry.get("status", "")),
                "keep_actor": True,
                "strict_retention_allowed": gate_pass,
                "strict_retention_decision": (
                    "allow_promotable_retention" if gate_pass else "block_current_gate_failed"
                ),
                "gate_pass": gate_pass,
                "ytd_return_pct": gate_row.get("ytd_return_pct"),
                "worst_wf_max_drawdown_pct": gate_row.get("worst_wf_max_drawdown_pct"),
                "wf_windows_present": gate_row.get("wf_windows_present", ""),
                "failures": failures,
                "notes_ref": str(entry.get("notes_ref", "")),
            }
        )
    return audit


def _evaluate_profile(
    *,
    root: Path,
    profile: dict[str, Any],
    ytd_row: pd.Series | None,
    reference: ReferenceBaseline,
    current_best_return_pct: float,
    max_walkforward_drawdown_pct: float,
    best_return_ratio: float,
) -> dict[str, Any]:
    profile_name = str(profile["profile"])
    wf_metrics = load_walkforward_metrics(root, profile_name)
    failures: list[str] = []
    for window in REQUIRED_WF_WINDOWS:
        if window not in wf_metrics:
            failures.append(f"missing_wf_window:{window}")

    threshold_label = _threshold_label(max_walkforward_drawdown_pct)
    for window, metrics in wf_metrics.items():
        if float(metrics["max_drawdown_pct"]) >= max_walkforward_drawdown_pct:
            failures.append(f"wf_max_drawdown_ge_{threshold_label}:{window}")

    ytd_return_column = _first_existing_column(ytd_row, RETURN_COLUMNS) if ytd_row is not None else None
    ytd_max_dd_column = _first_existing_column(ytd_row, MAX_DRAWDOWN_COLUMNS) if ytd_row is not None else None
    ytd_return = float(ytd_row[ytd_return_column]) if ytd_row is not None and ytd_return_column else None
    ytd_max_dd = float(ytd_row[ytd_max_dd_column]) if ytd_row is not None and ytd_max_dd_column else None
    required_best_return = current_best_return_pct * best_return_ratio
    if ytd_return is None:
        failures.append("missing_ytd_return")
    else:
        if ytd_return < reference.return_pct:
            failures.append("ytd_return_below_soxlbacktest_baseline")
        if ytd_return < required_best_return:
            failures.append("ytd_return_below_best_alignment")

    gate_pass = not failures
    return {
        "profile": profile_name,
        "status": profile.get("status"),
        "source_path": profile.get("output"),
        "ytd_return_pct": round(float(ytd_return), 6) if ytd_return is not None else None,
        "ytd_max_drawdown_pct": round(float(ytd_max_dd), 6) if ytd_max_dd is not None else None,
        "current_best_return_pct": round(current_best_return_pct, 6),
        "required_best_return_pct": round(required_best_return, 6),
        "required_soxlbacktest_return_pct": round(reference.return_pct, 6),
        "wf_windows_present": ",".join(sorted(wf_metrics)),
        "worst_wf_max_drawdown_pct": (
            round(max(float(item["max_drawdown_pct"]) for item in wf_metrics.values()), 6)
            if wf_metrics
            else None
        ),
        "gate_pass": gate_pass,
        "retention_decision": "retain_promotable" if gate_pass else "retain_as_evidence_only",
        "failures": failures,
    }


def scan_model_retention_gates(
    *,
    root: Path,
    max_walkforward_drawdown_pct: float = 20.0,
    best_return_ratio: float = 0.95,
) -> dict[str, Any]:
    root = Path(root)
    reference = load_reference_baseline(root)
    ytd = load_ytd_comparison(root)
    ytd_return_column = _first_existing_column(ytd, RETURN_COLUMNS)
    if ytd_return_column is None:
        raise ValueError(f"{COMPARISON_METRICS_PATH} must contain one of {', '.join(RETURN_COLUMNS)}")
    current_best_return_pct = float(ytd[ytd_return_column].max())
    ytd_by_profile = {str(row["profile"]): row for _, row in ytd.iterrows()}
    rows: list[dict[str, Any]] = [_reference_row(reference)]
    for profile in _load_replayable_profiles(root):
        profile_name = str(profile["profile"])
        rows.append(
            _evaluate_profile(
                root=root,
                profile=profile,
                ytd_row=ytd_by_profile.get(profile_name),
                reference=reference,
                current_best_return_pct=current_best_return_pct,
                max_walkforward_drawdown_pct=max_walkforward_drawdown_pct,
                best_return_ratio=best_return_ratio,
            )
        )
    passing_profiles = sorted(
        row["profile"]
        for row in rows
        if row["profile"] != PRIMARY_REFERENCE_PROFILE and row["gate_pass"]
    )
    registry_keep_actor_audit = _registry_keep_actor_audit(rows, _load_registry_keep_actor_entries(root))
    registry_keep_actor_violations = [
        row for row in registry_keep_actor_audit if not bool(row["strict_retention_allowed"])
    ]
    return {
        "summary": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "profiles_scanned": len(rows) - 1,
            "passing_profiles": passing_profiles,
            "passing_profile_count": len(passing_profiles),
            "registry_keep_actor_count": len(registry_keep_actor_audit),
            "registry_keep_actor_strict_violation_count": len(registry_keep_actor_violations),
            "registry_keep_actor_strict_violations": sorted(row["profile"] for row in registry_keep_actor_violations),
            "reference_profile": PRIMARY_REFERENCE_PROFILE,
            "reference_return_pct": round(reference.return_pct, 6),
            "reference_max_drawdown_pct": round(reference.max_drawdown_pct, 6),
            "current_best_return_pct": round(current_best_return_pct, 6),
            "best_return_ratio": float(best_return_ratio),
            "required_best_return_pct": round(current_best_return_pct * best_return_ratio, 6),
            "max_walkforward_drawdown_pct": float(max_walkforward_drawdown_pct),
            "required_wf_windows": list(REQUIRED_WF_WINDOWS),
        },
        "rows": rows,
        "registry_keep_actor_audit": registry_keep_actor_audit,
    }


def _csv_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        payload = dict(row)
        failures = payload.get("failures")
        if isinstance(failures, list):
            payload["failures"] = ";".join(str(item) for item in failures)
        result.append(payload)
    return result


def write_retention_gate_outputs(result: dict[str, Any], *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    pd.DataFrame(_csv_rows(rows)).to_csv(output_dir / "retention_gate.csv", index=False)
    registry_audit = result.get("registry_keep_actor_audit", [])
    pd.DataFrame(_csv_rows(registry_audit)).to_csv(output_dir / "registry_keep_actor_audit.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = result["summary"]
    passing = summary["passing_profiles"]
    conclusion = [
        "# Model Retention Walk-Forward Gate",
        "",
        f"- Reference baseline: `{summary['reference_profile']}` return `{summary['reference_return_pct']:.6f}%`, MaxDD `{summary['reference_max_drawdown_pct']:.6f}%`.",
        f"- Current best YTD model return: `{summary['current_best_return_pct']:.6f}%`; required alignment floor at ratio `{summary['best_return_ratio']:.2f}` is `{summary['required_best_return_pct']:.6f}%`.",
        f"- Walk-forward requirement: windows `{', '.join(summary['required_wf_windows'])}` all present and MaxDD strictly below `{summary['max_walkforward_drawdown_pct']:.2f}%`.",
        f"- Passing promotable profiles: `{', '.join(passing) if passing else 'none'}`.",
        f"- Registry keep_actor entries: `{summary.get('registry_keep_actor_count', 0)}`; keep_actor strict violations: `{summary.get('registry_keep_actor_strict_violation_count', 0)}`.",
        "",
        "This card is a retention/promotion audit. It does not delete actors or override production runtime protection.",
    ]
    (output_dir / "conclusion.md").write_text("\n".join(conclusion) + "\n", encoding="utf-8")
