from __future__ import annotations

"""Retention gate review for replayable SOXL models.

The gate is intentionally stricter than the historical runtime-protection
registry: executable actor retention is allowed only when the model satisfies
the current walk-forward/drawdown/return objective. Failed actors are downgraded
to summary-only evidence; actor files are not deleted by this review.
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
DEFAULT_MIN_WF_RELATIVE_RETURN_PCT = 30.0
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


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _comparison_evidence_profiles(ytd: pd.DataFrame) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in ytd.iterrows():
        profile = _string_value(row.get("profile")).strip()
        if not profile or profile == PRIMARY_REFERENCE_PROFILE or profile in seen:
            continue
        model_kind = _string_value(row.get("model_kind")).strip()
        status = _string_value(row.get("status")).strip()
        if model_kind == "runtime_composer" or status == "production_composer":
            continue
        source_path = _string_value(row.get("source_path")).strip()
        profiles.append(
            {
                "profile": profile,
                "status": status or "comparison_evidence",
                "output": source_path or f"runtime/model/card/{profile}/{profile}_soxl_daily.csv",
                "result": "comparison_evidence",
            }
        )
        seen.add(profile)
    return profiles


def _model_retention_scan_profiles(root: Path, ytd: pd.DataFrame) -> list[dict[str, Any]]:
    profiles = _load_replayable_profiles(root)
    if profiles:
        return profiles
    return _comparison_evidence_profiles(ytd)


def _load_registry_keep_actor_entries(root: Path) -> list[dict[str, Any]]:
    payload = _load_registry_payload(root)
    entries = payload.get("profiles", [])
    if not isinstance(entries, list):
        raise ValueError(f"{REGISTRY_PATH} must contain profiles list")
    return [
        entry
        for entry in entries
        if isinstance(entry, dict) and bool(entry.get("keep_actor", False)) and entry.get("name")
    ]


def _load_registry_payload(root: Path) -> dict[str, Any]:
    path = root / REGISTRY_PATH
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{REGISTRY_PATH} must contain mapping payload")
    return payload


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
        "min_wf_relative_return_pct": None,
        "missing_wf_window_count": 0,
        "wf_drawdown_gap_pct": None,
        "wf_relative_return_gap_pct": None,
        "best_return_gap_pct": None,
        "soxlbacktest_return_gap_pct": None,
        "gate_pass": True,
        "retention_decision": "keep_reference_baseline",
        "failures": [],
    }


def _registry_keep_actor_audit(
    rows: Iterable[dict[str, Any]],
    registry_entries: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
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
    min_walkforward_relative_return_pct: float,
    best_return_ratio: float,
) -> dict[str, Any]:
    profile_name = str(profile["profile"])
    wf_metrics = load_walkforward_metrics(root, profile_name)
    failures: list[str] = []
    for window in REQUIRED_WF_WINDOWS:
        if window not in wf_metrics:
            failures.append(f"missing_wf_window:{window}")
    missing_wf_window_count = sum(1 for window in REQUIRED_WF_WINDOWS if window not in wf_metrics)

    threshold_label = _threshold_label(max_walkforward_drawdown_pct)
    worst_wf_max_drawdown = (
        max(float(item["max_drawdown_pct"]) for item in wf_metrics.values())
        if wf_metrics
        else None
    )
    for window, metrics in wf_metrics.items():
        if float(metrics["max_drawdown_pct"]) >= max_walkforward_drawdown_pct:
            failures.append(f"wf_max_drawdown_ge_{threshold_label}:{window}")

    relative_threshold_label = _threshold_label(min_walkforward_relative_return_pct)
    min_wf_relative_return = (
        min(float(item["relative_return_pct"]) for item in wf_metrics.values())
        if wf_metrics
        else None
    )
    for window, metrics in wf_metrics.items():
        if float(metrics["relative_return_pct"]) < min_walkforward_relative_return_pct:
            failures.append(f"wf_relative_return_below_{relative_threshold_label}:{window}")

    ytd_return_column = _first_existing_column(ytd_row, RETURN_COLUMNS) if ytd_row is not None else None
    ytd_max_dd_column = _first_existing_column(ytd_row, MAX_DRAWDOWN_COLUMNS) if ytd_row is not None else None
    ytd_return = float(ytd_row[ytd_return_column]) if ytd_row is not None and ytd_return_column else None
    ytd_max_dd = float(ytd_row[ytd_max_dd_column]) if ytd_row is not None and ytd_max_dd_column else None
    required_best_return = current_best_return_pct * best_return_ratio
    best_return_gap = None
    soxlbacktest_return_gap = None
    if ytd_return is None:
        failures.append("missing_ytd_return")
    else:
        best_return_gap = max(0.0, required_best_return - ytd_return)
        soxlbacktest_return_gap = max(0.0, reference.return_pct - ytd_return)
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
        "worst_wf_max_drawdown_pct": round(worst_wf_max_drawdown, 6) if worst_wf_max_drawdown is not None else None,
        "min_wf_relative_return_pct": (
            round(min_wf_relative_return, 6) if min_wf_relative_return is not None else None
        ),
        "missing_wf_window_count": missing_wf_window_count,
        "wf_drawdown_gap_pct": (
            round(max(0.0, worst_wf_max_drawdown - max_walkforward_drawdown_pct), 6)
            if worst_wf_max_drawdown is not None
            else None
        ),
        "wf_relative_return_gap_pct": (
            round(max(0.0, min_walkforward_relative_return_pct - min_wf_relative_return), 6)
            if min_wf_relative_return is not None
            else None
        ),
        "best_return_gap_pct": round(best_return_gap, 6) if best_return_gap is not None else None,
        "soxlbacktest_return_gap_pct": (
            round(soxlbacktest_return_gap, 6) if soxlbacktest_return_gap is not None else None
        ),
        "gate_pass": gate_pass,
        "retention_decision": "retain_promotable" if gate_pass else "retain_as_evidence_only",
        "failures": failures,
    }


def _candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("profile") != PRIMARY_REFERENCE_PROFILE]


def _closest_wf_drawdown_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in _candidate_rows(rows)
        if row.get("worst_wf_max_drawdown_pct") is not None
    ]
    if not candidates:
        return None
    row = sorted(
        candidates,
        key=lambda item: (
            int(item.get("missing_wf_window_count") or 0),
            float(item.get("wf_drawdown_gap_pct") or 0.0),
            float(item.get("wf_relative_return_gap_pct") or 0.0),
            float(item.get("best_return_gap_pct") or 0.0),
        ),
    )[0]
    return {
        "profile": row["profile"],
        "worst_wf_max_drawdown_pct": row.get("worst_wf_max_drawdown_pct"),
        "wf_drawdown_gap_pct": row.get("wf_drawdown_gap_pct"),
        "min_wf_relative_return_pct": row.get("min_wf_relative_return_pct"),
        "wf_relative_return_gap_pct": row.get("wf_relative_return_gap_pct"),
        "missing_wf_window_count": row.get("missing_wf_window_count"),
        "ytd_return_pct": row.get("ytd_return_pct"),
        "best_return_gap_pct": row.get("best_return_gap_pct"),
    }


def _closest_ytd_return_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in _candidate_rows(rows)
        if row.get("ytd_return_pct") is not None
    ]
    if not candidates:
        return None
    row = sorted(
        candidates,
        key=lambda item: (
            float(item.get("best_return_gap_pct") or 0.0),
            int(item.get("missing_wf_window_count") or 0),
            float(item.get("wf_drawdown_gap_pct") or 0.0),
            float(item.get("wf_relative_return_gap_pct") or 0.0),
        ),
    )[0]
    return {
        "profile": row["profile"],
        "ytd_return_pct": row.get("ytd_return_pct"),
        "best_return_gap_pct": row.get("best_return_gap_pct"),
        "worst_wf_max_drawdown_pct": row.get("worst_wf_max_drawdown_pct"),
        "wf_drawdown_gap_pct": row.get("wf_drawdown_gap_pct"),
        "min_wf_relative_return_pct": row.get("min_wf_relative_return_pct"),
        "wf_relative_return_gap_pct": row.get("wf_relative_return_gap_pct"),
        "missing_wf_window_count": row.get("missing_wf_window_count"),
    }


def scan_model_retention_gates(
    *,
    root: Path,
    max_walkforward_drawdown_pct: float = 20.0,
    min_walkforward_relative_return_pct: float = DEFAULT_MIN_WF_RELATIVE_RETURN_PCT,
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
    for profile in _model_retention_scan_profiles(root, ytd):
        profile_name = str(profile["profile"])
        rows.append(
            _evaluate_profile(
                root=root,
                profile=profile,
                ytd_row=ytd_by_profile.get(profile_name),
                reference=reference,
                current_best_return_pct=current_best_return_pct,
                max_walkforward_drawdown_pct=max_walkforward_drawdown_pct,
                min_walkforward_relative_return_pct=min_walkforward_relative_return_pct,
                best_return_ratio=best_return_ratio,
            )
        )
    passing_profiles = sorted(
        row["profile"]
        for row in rows
        if row["profile"] != PRIMARY_REFERENCE_PROFILE and row["gate_pass"]
    )
    registry_payload = _load_registry_payload(root)
    registry_keep_actor_audit = _registry_keep_actor_audit(rows, _load_registry_keep_actor_entries(root))
    registry_keep_actor_violations = [
        row for row in registry_keep_actor_audit if not bool(row["strict_retention_allowed"])
    ]
    registry_retention_plan = plan_registry_retention_updates(registry_payload or {}, {"registry_keep_actor_audit": registry_keep_actor_audit})
    registry_disable_plan = [row for row in registry_retention_plan if row["action"] == "disable_keep_actor"]
    return {
        "summary": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "profiles_scanned": len(rows) - 1,
            "passing_profiles": passing_profiles,
            "passing_profile_count": len(passing_profiles),
            "registry_keep_actor_count": len(registry_keep_actor_audit),
            "registry_keep_actor_strict_violation_count": len(registry_keep_actor_violations),
            "registry_keep_actor_disable_count": len(registry_disable_plan),
            "registry_keep_actor_protected_exception_count": 0,
            "registry_keep_actor_strict_violations": sorted(
                row["profile"] for row in registry_keep_actor_violations
            ),
            "reference_profile": PRIMARY_REFERENCE_PROFILE,
            "reference_return_pct": round(reference.return_pct, 6),
            "reference_max_drawdown_pct": round(reference.max_drawdown_pct, 6),
            "current_best_return_pct": round(current_best_return_pct, 6),
            "best_return_ratio": float(best_return_ratio),
            "required_best_return_pct": round(current_best_return_pct * best_return_ratio, 6),
            "max_walkforward_drawdown_pct": float(max_walkforward_drawdown_pct),
            "min_walkforward_relative_return_pct": float(min_walkforward_relative_return_pct),
            "required_wf_windows": list(REQUIRED_WF_WINDOWS),
            "closest_wf_drawdown_candidate": _closest_wf_drawdown_candidate(rows),
            "closest_ytd_return_candidate": _closest_ytd_return_candidate(rows),
        },
        "rows": rows,
        "registry_keep_actor_audit": registry_keep_actor_audit,
        "registry_retention_plan": registry_retention_plan,
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


def _audit_by_profile(gate_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    audit = gate_result.get("registry_keep_actor_audit", [])
    if not isinstance(audit, list):
        return {}
    return {str(row.get("profile")): row for row in audit if isinstance(row, dict) and row.get("profile")}


def _normalized_failures(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value:
        return [part for part in value.split(";") if part]
    return []


def plan_registry_retention_updates(
    registry_payload: dict[str, Any],
    gate_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Plan registry keep_actor updates from the strict retention gate.

    The plan is intentionally strict: every failed `keep_actor` row is
    downgraded to summary-only when it fails the walk-forward/drawdown/return
    gate, regardless of lifecycle status. This keeps rollback evidence while
    preventing old protected statuses from masquerading as final retained
    executable models.
    """

    entries = registry_payload.get("profiles", [])
    if not isinstance(entries, list):
        raise ValueError("registry payload must contain profiles list")
    audit_by_profile = _audit_by_profile(gate_result)
    plan: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or not bool(entry.get("keep_actor", False)):
            continue
        profile = str(entry.get("name") or "").strip()
        if not profile:
            continue
        status = str(entry.get("status") or "").strip()
        audit = audit_by_profile.get(profile, {})
        failures = _normalized_failures(audit.get("failures"))
        strict_allowed = bool(audit.get("strict_retention_allowed", False))
        if strict_allowed:
            action = "keep_actor_allowed"
        else:
            action = "disable_keep_actor"
        plan.append(
            {
                "profile": profile,
                "status": status,
                "action": action,
                "strict_retention_allowed": strict_allowed,
                "strict_retention_decision": str(audit.get("strict_retention_decision") or ""),
                "failures": failures,
            }
        )
    return plan


def apply_registry_retention_updates(
    registry_payload: dict[str, Any],
    plan: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Apply strict-retention failures to a registry payload."""

    disable_by_profile = {
        str(row.get("profile")): row
        for row in plan
        if isinstance(row, dict) and row.get("action") == "disable_keep_actor" and row.get("profile")
    }
    updated: dict[str, Any] = dict(registry_payload)
    entries = registry_payload.get("profiles", [])
    if not isinstance(entries, list):
        raise ValueError("registry payload must contain profiles list")
    changed: list[str] = []
    new_entries: list[Any] = []
    for entry in entries:
        if not isinstance(entry, dict):
            new_entries.append(entry)
            continue
        profile = str(entry.get("name") or "")
        plan_row = disable_by_profile.get(profile)
        if plan_row is None:
            new_entries.append(dict(entry))
            continue
        failures = _normalized_failures(plan_row.get("failures"))
        new_entry = dict(entry)
        new_entry["keep_actor"] = False
        new_entry["keep_summary"] = True
        new_entry["wf_status"] = "blocked"
        reason = ";".join(failures) if failures else str(plan_row.get("strict_retention_decision") or "strict_retention_failed")
        gate_note = (
            "Strict retention gate blocked executable actor retention; "
            f"kept summary evidence only. Failures: {reason}"
        )
        existing_note = str(entry.get("wf_notes") or "").strip()
        new_entry["wf_notes"] = f"{existing_note} {gate_note}".strip()
        new_entries.append(new_entry)
        changed.append(profile)
    updated["profiles"] = new_entries
    return updated, changed


def write_registry_retention_updates(root: Path, gate_result: dict[str, Any]) -> list[str]:
    """Persist strict-retention registry metadata updates.

    This updates `keep_actor` rows only; it never deletes actor or summary
    artifacts. The returned list contains profiles changed to summary-only.
    """

    root = Path(root)
    registry_path = root / REGISTRY_PATH
    registry_payload = _load_registry_payload(root)
    updated, changed = apply_registry_retention_updates(
        registry_payload,
        gate_result.get("registry_retention_plan", []),
    )
    if changed:
        registry_path.write_text(
            yaml.safe_dump(updated, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return changed


def write_retention_gate_outputs(result: dict[str, Any], *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    pd.DataFrame(_csv_rows(rows)).to_csv(output_dir / "retention_gate.csv", index=False)
    registry_audit = result.get("registry_keep_actor_audit", [])
    pd.DataFrame(_csv_rows(registry_audit)).to_csv(output_dir / "registry_keep_actor_audit.csv", index=False)
    registry_plan = result.get("registry_retention_plan", [])
    pd.DataFrame(_csv_rows(registry_plan)).to_csv(output_dir / "registry_retention_plan.csv", index=False)
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
        f"- Walk-forward requirement: windows `{', '.join(summary['required_wf_windows'])}` all present, MaxDD strictly below `{summary['max_walkforward_drawdown_pct']:.2f}%`, and relative return at least `{summary.get('min_walkforward_relative_return_pct', DEFAULT_MIN_WF_RELATIVE_RETURN_PCT):.2f}pp` in every window.",
        f"- Passing promotable profiles: `{', '.join(passing) if passing else 'none'}`.",
        f"- Registry keep_actor entries: `{summary.get('registry_keep_actor_count', 0)}`; keep_actor strict violations: `{summary.get('registry_keep_actor_strict_violation_count', 0)}`.",
        f"- Registry actor disable plan: `{summary.get('registry_keep_actor_disable_count', 0)}`; protected exceptions: `0`.",
        "",
        "This card is a retention/promotion audit. Applying the registry plan downgrades every failed executable actor to summary-only evidence; it does not delete actor files.",
    ]
    (output_dir / "conclusion.md").write_text("\n".join(conclusion) + "\n", encoding="utf-8")
