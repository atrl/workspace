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


PRIMARY_REFERENCE_PROFILE = "soxl_backtest_policy_reimplementation_v1"
REQUIRED_WF_WINDOWS = ("wf_2020", "wf_2022", "wf_2023_2024")
REFERENCE_METRICS_PATH = (
    "runtime/production/card/soxl_backtest_policy_reimplementation_v1/legacy_comparison_metrics.csv"
)
COMPARISON_METRICS_PATH = (
    "runtime/production/card/current_model_ytd_june_comparison_v1/all_model_comparison_metrics.csv"
)
FRESHNESS_PATH = "dashboard_data/model_replay_freshness.json"
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
        "missing_wf_window_count": 0,
        "wf_drawdown_gap_pct": None,
        "best_return_gap_pct": None,
        "soxlbacktest_return_gap_pct": None,
        "gate_pass": True,
        "retention_decision": "keep_reference_baseline",
        "failures": [],
    }


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
        "missing_wf_window_count": missing_wf_window_count,
        "wf_drawdown_gap_pct": (
            round(max(0.0, worst_wf_max_drawdown - max_walkforward_drawdown_pct), 6)
            if worst_wf_max_drawdown is not None
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
            float(item.get("best_return_gap_pct") or 0.0),
        ),
    )[0]
    return {
        "profile": row["profile"],
        "worst_wf_max_drawdown_pct": row.get("worst_wf_max_drawdown_pct"),
        "wf_drawdown_gap_pct": row.get("wf_drawdown_gap_pct"),
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
        ),
    )[0]
    return {
        "profile": row["profile"],
        "ytd_return_pct": row.get("ytd_return_pct"),
        "best_return_gap_pct": row.get("best_return_gap_pct"),
        "worst_wf_max_drawdown_pct": row.get("worst_wf_max_drawdown_pct"),
        "wf_drawdown_gap_pct": row.get("wf_drawdown_gap_pct"),
        "missing_wf_window_count": row.get("missing_wf_window_count"),
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
    return {
        "summary": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "profiles_scanned": len(rows) - 1,
            "passing_profiles": passing_profiles,
            "passing_profile_count": len(passing_profiles),
            "reference_profile": PRIMARY_REFERENCE_PROFILE,
            "reference_return_pct": round(reference.return_pct, 6),
            "reference_max_drawdown_pct": round(reference.max_drawdown_pct, 6),
            "current_best_return_pct": round(current_best_return_pct, 6),
            "best_return_ratio": float(best_return_ratio),
            "required_best_return_pct": round(current_best_return_pct * best_return_ratio, 6),
            "max_walkforward_drawdown_pct": float(max_walkforward_drawdown_pct),
            "required_wf_windows": list(REQUIRED_WF_WINDOWS),
            "closest_wf_drawdown_candidate": _closest_wf_drawdown_candidate(rows),
            "closest_ytd_return_candidate": _closest_ytd_return_candidate(rows),
        },
        "rows": rows,
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value) if value is not None else False


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    try:
        if pd.isna(number):
            return None
    except (TypeError, ValueError):
        return None
    return number


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    return int(number)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, dict, set)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _failure_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        if not value.strip():
            return []
        separator = ";" if ";" in value else ","
        return [item.strip() for item in value.split(separator) if item.strip()]
    return []


def _wf_window_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def _rows_from_evidence(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows = evidence.get("rows")
    if isinstance(rows, pd.DataFrame):
        return rows.to_dict(orient="records")
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _failed_gate_reason(evidence: dict[str, Any], pass_key: str) -> str:
    failures = evidence.get("failures") or evidence.get("blocking_reasons") or [f"{pass_key}=false"]
    if isinstance(failures, list):
        return "; ".join(str(item) for item in failures)
    return str(failures)


def _strict_retention_row_status(
    evidence: dict[str, Any],
    *,
    profile_name: str | None,
    summary: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    profile = str(evidence.get("profile") or "")
    if profile_name is not None and evidence.get("profile") not in (None, profile_name):
        return False, f"strict retention gate profile mismatch: {evidence.get('profile')}"
    if (
        profile == PRIMARY_REFERENCE_PROFILE
        or str(evidence.get("status") or "") == "reference"
        or str(evidence.get("retention_decision") or "") == "keep_reference_baseline"
    ):
        return False, "SOXLBacktest reference baseline cannot be retained as actor"

    for flag in ("actor_retention_allowed", "promotion_allowed"):
        if flag in evidence and not _as_bool(evidence.get(flag)):
            return False, f"{flag}=false"

    pass_key = (
        "gate_pass"
        if "gate_pass" in evidence
        else "retention_gate_pass"
        if "retention_gate_pass" in evidence
        else None
    )
    if pass_key is None:
        return False, "strict retention gate did not contain a pass marker"
    if not _as_bool(evidence.get(pass_key)):
        return False, _failed_gate_reason(evidence, pass_key)

    required_fields = [
        "retention_decision",
        "failures",
        "ytd_return_pct",
        "required_best_return_pct",
        "required_soxlbacktest_return_pct",
        "wf_windows_present",
        "worst_wf_max_drawdown_pct",
        "missing_wf_window_count",
    ]
    missing_fields = [field for field in required_fields if field not in evidence or _is_missing(evidence.get(field))]
    if missing_fields:
        return False, "strict retention gate missing required fields: " + ", ".join(sorted(missing_fields))

    failures = _failure_list(evidence.get("failures"))
    if failures:
        return False, "; ".join(failures)

    if str(evidence.get("retention_decision") or "") != "retain_promotable":
        return False, f"retention_decision={evidence.get('retention_decision')} is not retain_promotable"

    missing_wf_window_count = _as_int(evidence.get("missing_wf_window_count"))
    if missing_wf_window_count is None:
        return False, "missing_wf_window_count is not numeric"
    if missing_wf_window_count != 0:
        return False, f"missing_wf_window_count={missing_wf_window_count}"

    required_windows = REQUIRED_WF_WINDOWS
    if summary and isinstance(summary.get("required_wf_windows"), list):
        required_windows = tuple(str(item) for item in summary["required_wf_windows"])
    wf_windows_present = _wf_window_set(evidence.get("wf_windows_present"))
    missing_windows = sorted(set(required_windows).difference(wf_windows_present))
    if missing_windows:
        return False, "missing_wf_window:" + ",".join(missing_windows)

    max_walkforward_drawdown_pct = _as_float(evidence.get("max_walkforward_drawdown_pct"))
    if max_walkforward_drawdown_pct is None and summary:
        max_walkforward_drawdown_pct = _as_float(summary.get("max_walkforward_drawdown_pct"))
    if max_walkforward_drawdown_pct is None:
        max_walkforward_drawdown_pct = 20.0

    worst_wf_max_drawdown_pct = _as_float(evidence.get("worst_wf_max_drawdown_pct"))
    if worst_wf_max_drawdown_pct is None:
        return False, "worst_wf_max_drawdown_pct is not numeric"
    if worst_wf_max_drawdown_pct >= max_walkforward_drawdown_pct:
        return False, f"wf_max_drawdown_ge_{_threshold_label(max_walkforward_drawdown_pct)}"

    ytd_return_pct = _as_float(evidence.get("ytd_return_pct"))
    required_best_return_pct = _as_float(evidence.get("required_best_return_pct"))
    required_soxlbacktest_return_pct = _as_float(evidence.get("required_soxlbacktest_return_pct"))
    if ytd_return_pct is None:
        return False, "ytd_return_pct is not numeric"
    if required_best_return_pct is None:
        return False, "required_best_return_pct is not numeric"
    if required_soxlbacktest_return_pct is None:
        return False, "required_soxlbacktest_return_pct is not numeric"
    if ytd_return_pct < required_soxlbacktest_return_pct:
        return False, "ytd_return_below_soxlbacktest_baseline"
    if ytd_return_pct < required_best_return_pct:
        return False, "ytd_return_below_best_alignment"

    return True, "strict retention gate passed"


def strict_retention_gate_status(
    evidence: dict[str, Any] | None,
    *,
    profile_name: str | None = None,
) -> tuple[bool, str]:
    """Validate explicit evidence before a checkpoint actor becomes reusable.

    Retention evidence can be either a row emitted by ``scan_model_retention_gates``
    or a summary/selection payload that names the passing profiles. Missing
    evidence is a hard block because validation-time checkpoints do not carry
    independent walk-forward/SOXLBacktest proof.
    """

    if not isinstance(evidence, dict):
        return False, "strict retention gate evidence missing"

    if profile_name is not None and evidence.get("profile") not in (None, profile_name):
        return False, f"strict retention gate profile mismatch: {evidence.get('profile')}"

    summary = evidence.get("summary")
    rows = _rows_from_evidence(evidence)
    if isinstance(summary, dict) and rows and profile_name is not None:
        for row in rows:
            if str(row.get("profile") or "") == profile_name:
                return _strict_retention_row_status(row, profile_name=profile_name, summary=summary)
        return False, f"profile {profile_name} is not in retention gate rows"

    if isinstance(summary, dict) and profile_name is not None:
        passing_profiles = summary.get("passing_profiles")
        if isinstance(passing_profiles, list) and profile_name in {str(item) for item in passing_profiles}:
            return False, f"strict retention gate row evidence missing for profile {profile_name}"
        return False, f"profile {profile_name} is not in retention gate passing_profiles"

    pass_key = (
        "gate_pass"
        if "gate_pass" in evidence
        else "retention_gate_pass"
        if "retention_gate_pass" in evidence
        else None
    )
    if pass_key is not None:
        return _strict_retention_row_status(evidence, profile_name=profile_name)

    for key in ("retention_gate_pass_candidates", "passing_profiles"):
        values = evidence.get(key)
        if isinstance(values, list) and profile_name is not None:
            if profile_name in {str(item) for item in values}:
                return False, f"strict retention gate row evidence missing for profile {profile_name}"
            return False, f"profile {profile_name} is not in {key}"

    return False, "strict retention gate did not contain a pass marker"


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
        "",
        "This card is a retention/promotion audit. It does not delete actors or override production runtime protection.",
    ]
    (output_dir / "conclusion.md").write_text("\n".join(conclusion) + "\n", encoding="utf-8")
