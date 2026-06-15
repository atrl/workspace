from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from src.soxl_quant.app.strict_retention_handoff import derive_strict_retention_handoff


REFERENCE_PROFILE = "soxl_backtest_policy_reimplementation_v1"
SOXLBACKTEST_QLIB_SCREEN = "soxlbacktest_qlib_walkforward_baseline_v1"


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
    if not isinstance(value, (dict, list, tuple)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value) if value is not None else False


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _strict_summary(strict_gate_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = strict_gate_payload.get("summary")
    if isinstance(summary, Mapping):
        return summary
    return strict_gate_payload


def _strict_rows(strict_gate_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = strict_gate_payload.get("rows")
    if isinstance(rows, pd.DataFrame):
        return rows.to_dict(orient="records")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        return [row for row in rows if isinstance(row, Mapping)]
    return []


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


def _primary_blocker(failures: list[str], *, gate_pass: bool) -> str:
    if gate_pass:
        return ""
    return failures[0] if failures else "strict_walkforward_gate_failed"


def _profile_next_mechanism(row: Mapping[str, Any], *, gate_pass: bool, is_reference: bool) -> str:
    if is_reference:
        return "score_trained_candidates_against_reference"
    if gate_pass:
        return "promotion_review_required"
    failures = _failure_list(row.get("failures"))
    if any(item.startswith("wf_max_drawdown_ge_20") for item in failures):
        return "strict_walkforward_training_objective"
    if any(item.startswith("missing_wf_window") for item in failures):
        return "complete_walkforward_evaluation"
    if any("ytd_return_below" in item for item in failures):
        return "return_alignment_training_objective"
    return "strict_walkforward_training_objective"


def _profile_row(row: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    profile = str(row.get("profile") or "")
    is_reference = profile == REFERENCE_PROFILE or str(row.get("status") or "") == "reference"
    gate_pass = _as_bool(row.get("gate_pass")) and not is_reference
    strict_passing = int(summary.get("passing_profile_count") or 0)
    promotion_allowed = gate_pass and strict_passing > 0
    failures = _failure_list(row.get("failures"))
    primary_blocker = "reference_baseline_comparison_only" if is_reference else _primary_blocker(
        failures,
        gate_pass=promotion_allowed,
    )
    return {
        "candidate_id": profile,
        "candidate_type": "reference_baseline" if is_reference else "profile",
        "source_path": row.get("source_path"),
        "status": row.get("status"),
        "evidence_level": "strict_walkforward_reference" if is_reference else "strict_walkforward_profile",
        "runnable_actor": False if is_reference else True,
        "diagnostic_visible_allowed": True,
        "retention_allowed": promotion_allowed,
        "promotion_allowed": promotion_allowed,
        "actor_retention_allowed": promotion_allowed,
        "training_allowed_next": False if promotion_allowed else not is_reference,
        "ppo_allowed_next": False,
        "primary_blocker": primary_blocker,
        "failures": failures,
        "required_next_mechanism": _profile_next_mechanism(row, gate_pass=promotion_allowed, is_reference=is_reference),
        "ytd_return_pct": _as_float(row.get("ytd_return_pct")),
        "ytd_max_drawdown_pct": _as_float(row.get("ytd_max_drawdown_pct")),
        "worst_wf_max_drawdown_pct": _as_float(row.get("worst_wf_max_drawdown_pct")),
        "missing_wf_window_count": int(row.get("missing_wf_window_count") or 0),
        "retention_decision": row.get("retention_decision"),
    }


def _soxlbacktest_screen_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    selected = str(summary.get("selected_buy_sell_reference") or "")
    selected_metrics = summary.get(selected) if selected else None
    if not isinstance(selected_metrics, Mapping):
        selected_metrics = {}
    return {
        "candidate_id": SOXLBACKTEST_QLIB_SCREEN,
        "candidate_type": "qlib_reference_screen",
        "source_path": "runtime/model_selection/card/soxlbacktest_qlib_walkforward_baseline_v1/summary.json",
        "status": "reference_screen",
        "evidence_level": "qlib_aligned_soxlbacktest_reference",
        "runnable_actor": False,
        "diagnostic_visible_allowed": True,
        "retention_allowed": False,
        "promotion_allowed": False,
        "actor_retention_allowed": False,
        "training_allowed_next": False,
        "ppo_allowed_next": False,
        "primary_blocker": str(summary.get("decision") or "soxlbacktest_reference_not_promotable"),
        "failures": _failure_list(selected_metrics.get("failures")),
        "required_next_mechanism": "score_trained_candidates_against_reference",
        "selected_buy_sell_reference": selected,
        "selected_risk_floor": summary.get("selected_risk_floor"),
        "ytd_return_pct": _as_float(selected_metrics.get("ytd_return_pct")),
        "worst_wf_max_drawdown_pct": _as_float(selected_metrics.get("worst_wf_max_drawdown_pct")),
    }


def _checkpoint_selector_row(
    *,
    checkpoint_selection: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not checkpoint_selection:
        return None
    experiment = str(checkpoint_selection.get("experiment") or checkpoint_selection.get("profile") or "checkpoint_selector")
    blockers = handoff.get("fold_blockers") if isinstance(handoff.get("fold_blockers"), list) else []
    first_blocker = blockers[0] if blockers and isinstance(blockers[0], Mapping) else {}
    return {
        "candidate_id": experiment,
        "candidate_type": "checkpoint_selector",
        "source_path": f"runtime/model_selection/card/{experiment}/selection.json",
        "status": "diagnostic",
        "evidence_level": "three_fold_checkpoint_selector",
        "runnable_actor": False,
        "diagnostic_visible_allowed": True,
        "retention_allowed": False,
        "promotion_allowed": _as_bool(checkpoint_selection.get("promotion_allowed")),
        "actor_retention_allowed": _as_bool(checkpoint_selection.get("actor_retention_allowed")),
        "training_allowed_next": _as_bool(checkpoint_selection.get("training_allowed_from_this_card")),
        "ppo_allowed_next": _as_bool(handoff.get("ppo_allowed_next")),
        "primary_blocker": str(first_blocker.get("blocker") or checkpoint_selection.get("decision") or ""),
        "failures": [str(item) for item in checkpoint_selection.get("failed_folds") or []],
        "required_next_mechanism": str(handoff.get("decision") or checkpoint_selection.get("next_mechanism") or ""),
        "primary_failed_fold": str(handoff.get("primary_failed_fold") or ""),
        "failed_fold_count": handoff.get("failed_fold_count", len(checkpoint_selection.get("failed_folds") or [])),
        "target_relative_return_pp": _as_float(checkpoint_selection.get("target_relative_return_pp")),
        "target_max_dd_pct": _as_float(checkpoint_selection.get("target_max_dd_pct")),
    }


def derive_strict_candidate_retention_filter(
    *,
    strict_gate_payload: Mapping[str, Any],
    soxlbacktest_summary: Mapping[str, Any] | None = None,
    checkpoint_selection: Mapping[str, Any] | None = None,
    checkpoint_selection_by_fold: pd.DataFrame | None = None,
    checkpoint_candidate_metrics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Classify strict retention, SOXLBacktest reference, and checkpoint evidence.

    The result is intentionally conservative: reference baselines remain visible
    for comparison, but only strict walk-forward gate pass profiles can retain
    an actor bundle.
    """

    summary = _strict_summary(strict_gate_payload)
    rows = [_profile_row(row, summary) for row in _strict_rows(strict_gate_payload)]
    if soxlbacktest_summary:
        rows.append(_soxlbacktest_screen_row(soxlbacktest_summary))

    handoff: dict[str, Any] = {}
    if checkpoint_selection:
        handoff = derive_strict_retention_handoff(
            strict_gate_summary=summary,
            checkpoint_selection_by_fold=checkpoint_selection_by_fold
            if checkpoint_selection_by_fold is not None
            else pd.DataFrame(),
            checkpoint_candidate_metrics=checkpoint_candidate_metrics
            if checkpoint_candidate_metrics is not None
            else pd.DataFrame(),
        )
        selector_row = _checkpoint_selector_row(
            checkpoint_selection=checkpoint_selection,
            handoff=handoff,
        )
        if selector_row is not None:
            rows.append(selector_row)

    promotable_profiles = [
        row
        for row in rows
        if row["candidate_type"] == "profile" and row["promotion_allowed"] and row["actor_retention_allowed"]
    ]
    actor_retention_allowed_count = sum(1 for row in rows if row["actor_retention_allowed"])
    decision = (
        "strict_candidate_retention_filter_promotion_review_required"
        if promotable_profiles
        else "strict_candidate_retention_filter_no_promotable_candidates"
    )
    result = {
        "summary": {
            "decision": decision,
            "reference_baseline": summary.get("reference_profile") or REFERENCE_PROFILE,
            "profiles_scanned": summary.get("profiles_scanned"),
            "passing_profile_count": int(summary.get("passing_profile_count") or 0),
            "promotable_profile_count": len(promotable_profiles),
            "actor_retention_allowed_count": actor_retention_allowed_count,
            "reference_visible_count": sum(
                1 for row in rows if row["candidate_type"] in {"reference_baseline", "qlib_reference_screen"}
            ),
            "ppo_allowed_next": bool(handoff.get("ppo_allowed_next", False)),
            "next_mechanism": handoff.get("decision") or "strict_walkforward_training_objective",
            "max_walkforward_drawdown_pct": summary.get("max_walkforward_drawdown_pct"),
            "required_best_return_pct": summary.get("required_best_return_pct"),
            "required_wf_windows": [str(item) for item in summary.get("required_wf_windows") or []],
        },
        "rows": rows,
        "handoff": handoff,
    }
    return _json_safe(result)


def render_strict_candidate_retention_filter(result: Mapping[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), Mapping) else {}
    lines = [
        "# Strict Candidate Retention Filter",
        "",
        f"- decision: `{summary.get('decision', '')}`",
        f"- reference_baseline: `{summary.get('reference_baseline', '')}`",
        f"- promotable_profile_count: `{summary.get('promotable_profile_count', 0)}`",
        f"- actor_retention_allowed_count: `{summary.get('actor_retention_allowed_count', 0)}`",
        f"- ppo_allowed_next: `{summary.get('ppo_allowed_next', False)}`",
        "",
        "## Candidate Rows",
        "",
    ]
    for row in result.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "- "
            f"`{row.get('candidate_id')}` "
            f"({row.get('candidate_type')}): "
            f"promotion={row.get('promotion_allowed')}, "
            f"actor_retention={row.get('actor_retention_allowed')}, "
            f"blocker=`{row.get('primary_blocker', '')}`"
        )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "derive_strict_candidate_retention_filter",
    "render_strict_candidate_retention_filter",
]
