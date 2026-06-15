from __future__ import annotations

"""Final retention contract for walk-forward-qualified SOXL actors."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .model_retention_gate import (
    PRIMARY_REFERENCE_PROFILE,
    load_reference_baseline,
    strict_retention_gate_status,
)


REFERENCE_RETURN_TOLERANCE = 1e-6


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


def _summary(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = payload.get("summary")
    return summary if isinstance(summary, Mapping) else payload


def _rows(payload: Mapping[str, Any], key: str = "rows") -> list[Mapping[str, Any]]:
    rows = payload.get(key)
    if isinstance(rows, pd.DataFrame):
        return rows.to_dict(orient="records")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        return [row for row in rows if isinstance(row, Mapping)]
    return []


def _candidate_filter_rows(payload: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return {}
    return {
        str(row.get("candidate_id") or row.get("profile") or ""): row
        for row in _rows(payload)
        if row.get("candidate_type") in {None, "profile"}
    }


def _latest_reference_return(latest_reference_baseline: Mapping[str, Any] | None) -> float | None:
    if not isinstance(latest_reference_baseline, Mapping):
        return None
    for key in ("return_pct", "new_return_pct", "ytd_return_pct"):
        value = _as_float(latest_reference_baseline.get(key))
        if value is not None:
            return value
    return None


def _reference_freshness_blockers(
    *,
    strict_gate_summary: Mapping[str, Any],
    latest_reference_baseline: Mapping[str, Any] | None,
) -> list[str]:
    strict_reference_return = _as_float(strict_gate_summary.get("reference_return_pct"))
    latest_reference_return = _latest_reference_return(latest_reference_baseline)
    if strict_reference_return is None or latest_reference_return is None:
        return []
    if abs(strict_reference_return - latest_reference_return) > REFERENCE_RETURN_TOLERANCE:
        return ["stale_reference_return_pct"]
    return []


def derive_final_walkforward_retention_contract(
    *,
    strict_gate_payload: Mapping[str, Any],
    strict_candidate_filter_payload: Mapping[str, Any] | None,
    latest_reference_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine strict WF gate, candidate filter, and latest reference freshness.

    A final-retained actor must pass the strict walk-forward gate and also be
    allowed by the candidate-retention filter. SOXLBacktest is kept as a
    reference baseline only, never as a runnable actor.
    """

    strict_summary = _summary(strict_gate_payload)
    filter_summary = _summary(strict_candidate_filter_payload or {})
    filter_rows = _candidate_filter_rows(strict_candidate_filter_payload)
    global_blockers = _reference_freshness_blockers(
        strict_gate_summary=strict_summary,
        latest_reference_baseline=latest_reference_baseline,
    )
    if strict_candidate_filter_payload is None:
        global_blockers.append("missing_strict_candidate_retention_filter")

    result_rows: list[dict[str, Any]] = []
    for source_row in _rows(strict_gate_payload):
        row = dict(source_row)
        profile = str(row.get("profile") or "")
        is_reference = (
            profile == PRIMARY_REFERENCE_PROFILE
            or str(row.get("status") or "") == "reference"
            or str(row.get("retention_decision") or "") == "keep_reference_baseline"
        )
        strict_gate_ok, strict_gate_reason = strict_retention_gate_status(row, profile_name=profile)

        blockers = list(global_blockers)
        actor_retention_allowed = False
        promotion_allowed = False
        filter_blocker = ""
        filter_row = filter_rows.get(profile)
        if not blockers:
            if not strict_gate_ok:
                blockers.append(strict_gate_reason)
            elif not is_reference:
                if filter_row is None:
                    filter_blocker = "missing_strict_candidate_filter_row"
                    blockers.append(filter_blocker)
                else:
                    actor_retention_allowed = _as_bool(filter_row.get("actor_retention_allowed"))
                    promotion_allowed = _as_bool(filter_row.get("promotion_allowed"))
                    if not actor_retention_allowed:
                        filter_blocker = "strict_candidate_filter_actor_retention_false"
                        blockers.append(filter_blocker)

        result_rows.append(
            {
                "profile": profile,
                "status": row.get("status"),
                "source_path": row.get("source_path"),
                "strict_gate_pass": strict_gate_ok,
                "strict_gate_reason": strict_gate_reason,
                "promotion_allowed": promotion_allowed,
                "actor_retention_allowed": actor_retention_allowed,
                "final_model_retained": bool(strict_gate_ok and actor_retention_allowed and not blockers),
                "blockers": blockers,
                "filter_blocker": filter_blocker,
                "retention_decision": row.get("retention_decision"),
                "ytd_return_pct": _as_float(row.get("ytd_return_pct")),
                "required_best_return_pct": _as_float(row.get("required_best_return_pct")),
                "required_soxlbacktest_return_pct": _as_float(row.get("required_soxlbacktest_return_pct")),
                "worst_wf_max_drawdown_pct": _as_float(row.get("worst_wf_max_drawdown_pct")),
                "missing_wf_window_count": row.get("missing_wf_window_count"),
                "wf_windows_present": row.get("wf_windows_present"),
            }
        )

    final_retained = [row for row in result_rows if row["final_model_retained"]]
    decision = (
        "final_walkforward_retention_contract_ready"
        if final_retained and not global_blockers
        else "final_walkforward_retention_contract_no_retained_models"
    )
    summary = {
        "decision": decision,
        "reference_baseline": strict_summary.get("reference_profile") or PRIMARY_REFERENCE_PROFILE,
        "reference_return_pct": strict_summary.get("reference_return_pct"),
        "latest_reference_return_pct": _latest_reference_return(latest_reference_baseline),
        "reference_is_current": not bool(global_blockers),
        "global_blockers": global_blockers,
        "strict_passing_profile_count": int(strict_summary.get("passing_profile_count") or 0),
        "candidate_filter_actor_retention_allowed_count": int(
            filter_summary.get("actor_retention_allowed_count") or 0
        ),
        "final_retained_count": len(final_retained),
        "final_retained_profiles": sorted(str(row["profile"]) for row in final_retained),
        "current_best_return_pct": strict_summary.get("current_best_return_pct"),
        "required_best_return_pct": strict_summary.get("required_best_return_pct"),
        "max_walkforward_drawdown_pct": strict_summary.get("max_walkforward_drawdown_pct"),
        "required_wf_windows": [str(item) for item in strict_summary.get("required_wf_windows") or []],
    }
    return {"summary": summary, "rows": result_rows}


def load_latest_reference_baseline(root: Path) -> dict[str, Any]:
    reference = load_reference_baseline(root)
    return {
        "return_pct": reference.return_pct,
        "max_drawdown_pct": reference.max_drawdown_pct,
        "source_path": reference.source_path,
    }


def _csv_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    csv_rows = []
    for row in rows:
        payload = dict(row)
        blockers = payload.get("blockers")
        if isinstance(blockers, list):
            payload["blockers"] = ";".join(str(item) for item in blockers)
        csv_rows.append(payload)
    return csv_rows


def render_final_walkforward_retention_contract(result: Mapping[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), Mapping) else {}
    lines = [
        "# Final Walk-Forward Retention Contract",
        "",
        f"- decision: `{summary.get('decision', '')}`",
        f"- final_retained_count: `{summary.get('final_retained_count', 0)}`",
        f"- reference_baseline: `{summary.get('reference_baseline', '')}`",
        f"- reference_is_current: `{summary.get('reference_is_current', False)}`",
        f"- strict_passing_profile_count: `{summary.get('strict_passing_profile_count', 0)}`",
        "",
        "## Rows",
        "",
    ]
    for row in result.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        blockers = row.get("blockers")
        blocker_text = ";".join(str(item) for item in blockers) if isinstance(blockers, list) else str(blockers or "")
        lines.append(
            "- "
            f"`{row.get('profile')}`: "
            f"final_retained={row.get('final_model_retained')}, "
            f"actor_retention={row.get('actor_retention_allowed')}, "
            f"blockers=`{blocker_text}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_final_walkforward_retention_contract(result: Mapping[str, Any], *, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecars = {
        "final_retention_contract_csv": output_dir / "final_retention_contract.csv",
        "summary_json": output_dir / "summary.json",
        "markdown": output_dir / "final_walkforward_retention_contract.md",
    }
    pd.DataFrame(_csv_rows(result.get("rows", []))).to_csv(
        sidecars["final_retention_contract_csv"],
        index=False,
    )
    sidecars["summary_json"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecars["markdown"].write_text(
        render_final_walkforward_retention_contract(result),
        encoding="utf-8",
    )
    return sidecars


__all__ = [
    "derive_final_walkforward_retention_contract",
    "load_latest_reference_baseline",
    "render_final_walkforward_retention_contract",
    "write_final_walkforward_retention_contract",
]
