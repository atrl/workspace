from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any


REFERENCE_PROFILE = "soxl_backtest_policy_reimplementation_v1"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def _int(value: Any, default: int = 0) -> int:
    number = _float(value)
    return default if number is None else int(number)


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "passed", "pass"}
    return bool(value)


def _summary_payload(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not payload:
        return {}
    summary = payload.get("summary")
    return summary if isinstance(summary, Mapping) else payload


def _row_key(row: Mapping[str, Any]) -> str:
    return str(row.get("profile") or row.get("candidate_id") or "")


def _failure_list(row: Mapping[str, Any]) -> list[str]:
    failures = row.get("failures")
    if isinstance(failures, str):
        return [item for item in failures.split(";") if item]
    if isinstance(failures, Iterable) and not isinstance(failures, (bytes, dict)):
        return [str(item) for item in failures if str(item)]
    return []


def _is_reference_row(row: Mapping[str, Any], reference_profile: str) -> bool:
    profile = _row_key(row)
    decision = str(row.get("retention_decision") or "")
    return profile == reference_profile or decision == "keep_reference_baseline"


def _required_evidence_for_actor(row: Mapping[str, Any], role: str) -> list[str]:
    evidence = [
        "official_walkforward_gate",
        "worst_wf_maxdd_below_20",
        "ytd_return_above_best_alignment",
        "soxlbacktest_reference_alignment",
    ]
    if role == "complete_official_walkforward_before_actor_retention":
        evidence.insert(0, "complete_missing_wf_windows")
    if role == "train_drawdown_drift_and_return_alignment":
        evidence.append("drawdown_drift_stop_screen")
    return evidence


def _role_for_actor(row: Mapping[str, Any]) -> str:
    missing = _int(row.get("missing_wf_window_count"))
    drawdown_gap = _float(row.get("wf_drawdown_gap_pct"), 0.0) or 0.0
    return_gap = _float(row.get("best_return_gap_pct"), 0.0) or 0.0
    if missing > 0:
        return "complete_official_walkforward_before_actor_retention"
    if drawdown_gap > 0 and return_gap > 0:
        return "train_drawdown_drift_and_return_alignment"
    if drawdown_gap > 0:
        return "train_drawdown_control_objective"
    if return_gap > 0:
        return "train_return_alignment_without_drawdown_regression"
    return "promotion_review_candidate"


def _actor_priority(row: Mapping[str, Any]) -> tuple[int, float, float, str]:
    missing = _int(row.get("missing_wf_window_count"))
    drawdown_gap = _float(row.get("wf_drawdown_gap_pct"), 999.0)
    return_gap = _float(row.get("best_return_gap_pct"), 999.0)
    ytd = _float(row.get("ytd_return_pct"), 0.0) or 0.0
    complete_group = 0 if missing == 0 else 1
    score = (drawdown_gap or 0.0) * 10.0 + max(return_gap or 0.0, 0.0)
    if missing > 0:
        score = 500.0 + missing * 100.0 + max(return_gap or 0.0, 0.0)
    return complete_group, score, -ytd, _row_key(row)


def _actor_candidate_row(
    *,
    row: Mapping[str, Any],
    strict_gate_summary: Mapping[str, Any],
    candidate_filter_summary: Mapping[str, Any],
) -> dict[str, Any]:
    profile = _row_key(row)
    role = _role_for_actor(row)
    ytd_return = _float(row.get("ytd_return_pct"))
    required_best = _float(row.get("required_best_return_pct"), _float(strict_gate_summary.get("required_best_return_pct")))
    required_soxlbacktest = _float(
        row.get("required_soxlbacktest_return_pct"),
        _float(strict_gate_summary.get("reference_return_pct")),
    )
    missing = _int(row.get("missing_wf_window_count"))
    drawdown_gap = _float(row.get("wf_drawdown_gap_pct"))
    return_gap = _float(row.get("best_return_gap_pct"))
    failures = _failure_list(row)
    next_mechanism = str(
        candidate_filter_summary.get("next_mechanism")
        or "train_strict_retention_multi_objective_no_promotion"
    )
    primary_blocker = "missing_wf_windows" if missing > 0 else "drawdown_and_return_gap"
    if drawdown_gap and not return_gap:
        primary_blocker = "drawdown_gap"
    elif return_gap and not drawdown_gap:
        primary_blocker = "return_gap"
    return {
        "candidate_id": profile,
        "candidate_type": "trained_actor",
        "queue_role": role,
        "runnable_actor": True,
        "promotion_allowed": False,
        "actor_retention_allowed": False,
        "training_allowed_next": True,
        "screen_only": False,
        "primary_reason": "closest_complete_walkforward_gap" if missing == 0 else "return_candidate_missing_walkforward",
        "primary_blocker": primary_blocker,
        "next_mechanism": next_mechanism,
        "source_path": row.get("source_path"),
        "ytd_return_pct": ytd_return,
        "required_best_return_pct": required_best,
        "return_gap_pct": return_gap,
        "required_soxlbacktest_return_pct": required_soxlbacktest,
        "soxlbacktest_return_gap_pct": _float(row.get("soxlbacktest_return_gap_pct")),
        "worst_wf_max_drawdown_pct": _float(row.get("worst_wf_max_drawdown_pct")),
        "max_walkforward_drawdown_pct": _float(strict_gate_summary.get("max_walkforward_drawdown_pct"), 20.0),
        "wf_drawdown_gap_pct": drawdown_gap,
        "missing_wf_window_count": missing,
        "failures": failures,
        "required_evidence": _required_evidence_for_actor(row, role),
        "stop_condition": [
            "no_actor_retention_until_final_walkforward_contract_passes",
            "do_not_promote_without_all_required_wf_windows",
        ],
    }


def _reference_candidate_row(
    *,
    row: Mapping[str, Any] | None,
    strict_gate_summary: Mapping[str, Any],
    soxlbacktest_summary: Mapping[str, Any],
) -> dict[str, Any]:
    row = row or {}
    return {
        "candidate_id": str(strict_gate_summary.get("reference_profile") or REFERENCE_PROFILE),
        "candidate_type": "reference_baseline",
        "queue_role": "buy_sell_reference_baseline",
        "runnable_actor": False,
        "promotion_allowed": False,
        "actor_retention_allowed": False,
        "training_allowed_next": False,
        "screen_only": False,
        "primary_reason": "qlib_aligned_soxlbacktest_buy_sell_reference",
        "primary_blocker": "reference_baseline_not_actor",
        "next_mechanism": "compare_trained_candidates_against_buy_sell_reference",
        "source_path": row.get("source_path"),
        "ytd_return_pct": _float(row.get("ytd_return_pct"), _float(strict_gate_summary.get("reference_return_pct"))),
        "required_best_return_pct": _float(strict_gate_summary.get("required_best_return_pct")),
        "return_gap_pct": None,
        "required_soxlbacktest_return_pct": _float(strict_gate_summary.get("reference_return_pct")),
        "soxlbacktest_return_gap_pct": None,
        "worst_wf_max_drawdown_pct": None,
        "max_walkforward_drawdown_pct": _float(strict_gate_summary.get("max_walkforward_drawdown_pct"), 20.0),
        "wf_drawdown_gap_pct": None,
        "missing_wf_window_count": _int(row.get("missing_wf_window_count")),
        "failures": _failure_list(row),
        "selected_buy_sell_reference": soxlbacktest_summary.get("selected_buy_sell_reference"),
        "selected_risk_floor": soxlbacktest_summary.get("selected_risk_floor"),
        "required_evidence": [
            "compare_buy_sell_features",
            "common_window_rebased_nav",
            "reference_only_not_actor_retention",
        ],
        "stop_condition": ["do_not_promote_reference_baseline_as_actor"],
    }


def _screen_candidate_row(retraining_handoff_summary: Mapping[str, Any]) -> dict[str, Any] | None:
    if not retraining_handoff_summary:
        return None
    next_action = str(retraining_handoff_summary.get("next_action") or "")
    if not next_action:
        return None
    return {
        "candidate_id": "short_foldlocal_drawdown_drift_training_screen",
        "candidate_type": "training_screen",
        "queue_role": "screen_only_no_promotion",
        "runnable_actor": False,
        "promotion_allowed": False,
        "actor_retention_allowed": False,
        "training_allowed_next": True,
        "screen_only": True,
        "primary_reason": str(retraining_handoff_summary.get("decision") or "final_retraining_handoff"),
        "primary_blocker": "must_pass_official_walkforward_gate_after_screen",
        "next_mechanism": "short_foldlocal_drawdown_drift_reward_screen",
        "next_action": next_action,
        "source_path": "runtime/model_selection/card/final_walkforward_retraining_handoff_v1/summary.json",
        "required_evidence": [
            "foldlocal_screen_metrics",
            "official_walkforward_gate",
            "final_walkforward_retention_contract",
        ],
        "stop_condition": [
            "screen_output_is_not_promotable_without_official_wf_gate",
            "no_actor_retention_until_final_walkforward_contract_passes",
        ],
    }


def derive_strict_retraining_candidate_queue(
    *,
    strict_gate_summary: Mapping[str, Any],
    retention_gate_rows: Iterable[Mapping[str, Any]],
    final_contract_summary: Mapping[str, Any] | None = None,
    candidate_filter_summary: Mapping[str, Any] | None = None,
    soxlbacktest_summary: Mapping[str, Any] | None = None,
    retraining_handoff_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a no-promotion queue for the next strict walk-forward retraining loop."""

    strict_gate_summary = _summary_payload(strict_gate_summary)
    final_contract_summary = _summary_payload(final_contract_summary)
    candidate_filter_summary = _summary_payload(candidate_filter_summary)
    soxlbacktest_summary = _summary_payload(soxlbacktest_summary)
    retraining_handoff_summary = _summary_payload(retraining_handoff_summary)

    reference_profile = str(strict_gate_summary.get("reference_profile") or REFERENCE_PROFILE)
    rows = list(retention_gate_rows)
    actor_rows = [row for row in rows if not _is_reference_row(row, reference_profile)]
    reference_row = next((row for row in rows if _is_reference_row(row, reference_profile)), None)

    queue: list[dict[str, Any]] = [
        _actor_candidate_row(
            row=row,
            strict_gate_summary=strict_gate_summary,
            candidate_filter_summary=candidate_filter_summary,
        )
        for row in sorted(actor_rows, key=_actor_priority)
    ]
    if reference_row or soxlbacktest_summary:
        queue.append(
            _reference_candidate_row(
                row=reference_row,
                strict_gate_summary=strict_gate_summary,
                soxlbacktest_summary=soxlbacktest_summary,
            )
        )
    screen_row = _screen_candidate_row(retraining_handoff_summary)
    if screen_row is not None:
        queue.append(screen_row)

    for rank, row in enumerate(queue, start=1):
        row["queue_rank"] = rank

    final_retained_count = _int(final_contract_summary.get("final_retained_count"))
    promotion_allowed_count = sum(1 for row in queue if _bool(row.get("promotion_allowed")))
    actor_retention_allowed_count = sum(1 for row in queue if _bool(row.get("actor_retention_allowed")))
    top_actor = next((row for row in queue if row["candidate_type"] == "trained_actor"), None)
    decision = (
        "strict_retraining_candidate_queue_has_retained_model"
        if final_retained_count > 0
        else "strict_retraining_candidate_queue_ready_no_promotion"
    )
    required_gates = [
        {
            "gate": "final_walkforward_retention_contract",
            "required": True,
            "current": final_retained_count,
            "status": "passed" if final_retained_count > 0 else "blocked",
        },
        {
            "gate": "strict_walkforward_profile_count",
            "required": 1,
            "current": _int(strict_gate_summary.get("passing_profile_count")),
            "status": "passed" if _int(strict_gate_summary.get("passing_profile_count")) > 0 else "blocked",
        },
        {
            "gate": "max_walkforward_drawdown_pct",
            "required": _float(strict_gate_summary.get("max_walkforward_drawdown_pct"), 20.0),
            "current": top_actor.get("worst_wf_max_drawdown_pct") if top_actor else None,
            "status": "blocked",
        },
        {
            "gate": "required_best_return_pct",
            "required": _float(strict_gate_summary.get("required_best_return_pct")),
            "current": top_actor.get("ytd_return_pct") if top_actor else None,
            "status": "blocked",
        },
    ]
    return _json_safe(
        {
            "summary": {
                "decision": decision,
                "final_retained_count": final_retained_count,
                "strict_passing_profile_count": _int(strict_gate_summary.get("passing_profile_count")),
                "candidate_filter_actor_retention_allowed_count": _int(
                    final_contract_summary.get("candidate_filter_actor_retention_allowed_count")
                ),
                "promotion_allowed_count": promotion_allowed_count,
                "actor_retention_allowed_count": actor_retention_allowed_count,
                "queue_count": len(queue),
                "top_candidate_id": top_actor["candidate_id"] if top_actor else None,
                "reference_baseline": reference_profile,
                "reference_return_pct": _float(strict_gate_summary.get("reference_return_pct")),
                "current_best_return_pct": _float(strict_gate_summary.get("current_best_return_pct")),
                "required_best_return_pct": _float(strict_gate_summary.get("required_best_return_pct")),
                "max_walkforward_drawdown_pct": _float(strict_gate_summary.get("max_walkforward_drawdown_pct"), 20.0),
                "required_wf_windows": [str(item) for item in strict_gate_summary.get("required_wf_windows") or []],
                "next_action": retraining_handoff_summary.get("next_action")
                or candidate_filter_summary.get("next_mechanism"),
            },
            "candidate_queue": queue,
            "required_gates": required_gates,
        }
    )


def render_strict_retraining_candidate_queue(result: Mapping[str, Any]) -> str:
    summary = _summary_payload(result.get("summary") if isinstance(result, Mapping) else {})
    rows = result.get("candidate_queue", []) if isinstance(result, Mapping) else []
    required_gates = result.get("required_gates", []) if isinstance(result, Mapping) else []
    lines = [
        "# Strict Retraining Candidate Queue",
        "",
        f"- Decision: `{summary.get('decision')}`",
        f"- Final retained models: `{summary.get('final_retained_count')}`",
        f"- Top candidate: `{summary.get('top_candidate_id')}`",
        f"- Reference baseline: `{summary.get('reference_baseline')}`",
        f"- Required best return: `{summary.get('required_best_return_pct')}`",
        f"- Max WF drawdown: `{summary.get('max_walkforward_drawdown_pct')}`",
        "",
        "## Queue",
        "",
        "| Rank | Candidate | Type | Role | Promotion | Blocker | Evidence |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        evidence = ", ".join(str(item) for item in row.get("required_evidence") or [])
        lines.append(
            "| {rank} | `{candidate}` | {candidate_type} | {role} | {promotion} | {blocker} | {evidence} |".format(
                rank=row.get("queue_rank"),
                candidate=row.get("candidate_id"),
                candidate_type=row.get("candidate_type"),
                role=row.get("queue_role"),
                promotion=row.get("promotion_allowed"),
                blocker=row.get("primary_blocker"),
                evidence=evidence,
            )
        )
    lines.extend(["", "## Required Gates", "", "| Gate | Required | Current | Status |", "| --- | ---: | ---: | --- |"])
    for gate in required_gates:
        if not isinstance(gate, Mapping):
            continue
        lines.append(
            f"| `{gate.get('gate')}` | {gate.get('required')} | {gate.get('current')} | {gate.get('status')} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                "No actor can be retained until the final walk-forward contract passes. "
                "Use the first trained-actor rows as diagnostic retraining targets and keep "
                "SOXLBacktest as a reference baseline only."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_strict_retraining_candidate_queue(
    result: Mapping[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    queue_path = output_dir / "candidate_queue.csv"
    gates_path = output_dir / "required_gates.csv"
    markdown_path = output_dir / "strict_retraining_candidate_queue.md"

    summary = result.get("summary", {}) if isinstance(result, Mapping) else {}
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(result.get("candidate_queue", [])).to_csv(queue_path, index=False)
    pd.DataFrame(result.get("required_gates", [])).to_csv(gates_path, index=False)
    markdown_path.write_text(render_strict_retraining_candidate_queue(result), encoding="utf-8")
    return {
        "summary_json": summary_path,
        "candidate_queue_csv": queue_path,
        "required_gates_csv": gates_path,
        "markdown": markdown_path,
    }


__all__ = [
    "derive_strict_retraining_candidate_queue",
    "render_strict_retraining_candidate_queue",
    "write_strict_retraining_candidate_queue",
]
