from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


TARGET_RELATIVE_RETURN_PP = 30.0
PPO_MAX_DD_DRIFT_STOP_PP = 2.0


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
    if not isinstance(value, (list, tuple, dict)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def _bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _required_folds(strict_gate_summary: Mapping[str, Any]) -> list[str]:
    return [str(fold) for fold in strict_gate_summary.get("required_wf_windows") or []]


def _selection_row_by_fold(checkpoint_selection_by_fold: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    if checkpoint_selection_by_fold.empty:
        return {}
    return {
        str(row["fold"]): row
        for row in checkpoint_selection_by_fold.to_dict(orient="records")
        if row.get("fold")
    }


def _ppo_drift_for_fold(
    checkpoint_candidate_metrics: pd.DataFrame,
    fold: str,
    selected_relative_return: float,
) -> dict[str, Any]:
    if checkpoint_candidate_metrics.empty or "fold" not in checkpoint_candidate_metrics.columns:
        return {
            "ppo_max_dd_drift_pct": 0.0,
            "ppo_relative_improvement_pp": 0.0,
            "ppo_drift_candidate": "",
            "ppo_drift_stop": False,
        }
    frame = checkpoint_candidate_metrics[checkpoint_candidate_metrics["fold"].astype(str).eq(fold)].copy()
    if frame.empty:
        return {
            "ppo_max_dd_drift_pct": 0.0,
            "ppo_relative_improvement_pp": 0.0,
            "ppo_drift_candidate": "",
            "ppo_drift_stop": False,
        }
    if "candidate_kind" in frame.columns:
        frame = frame[~frame["candidate_kind"].astype(str).eq("bc_initialization")]
    else:
        frame = frame[~frame["candidate"].astype(str).eq("bc_initialization")]
    if frame.empty:
        return {
            "ppo_max_dd_drift_pct": 0.0,
            "ppo_relative_improvement_pp": 0.0,
            "ppo_drift_candidate": "",
            "ppo_drift_stop": False,
        }
    sort_cols = [column for column in ("test_relative_return", "relative_delta_vs_bc") if column in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols, ascending=False)
    best = frame.iloc[0].to_dict()
    max_dd_drift = _float(best.get("max_dd_delta_vs_bc"), 0.0)
    relative = _float(best.get("test_relative_return"), selected_relative_return)
    relative_improvement = relative - selected_relative_return
    return {
        "ppo_max_dd_drift_pct": round(max_dd_drift, 6),
        "ppo_relative_improvement_pp": round(relative_improvement, 6),
        "ppo_drift_candidate": str(best.get("candidate", "")),
        "ppo_drift_stop": max_dd_drift > PPO_MAX_DD_DRIFT_STOP_PP and relative_improvement > 0.0,
    }


def _blocker_for_row(row: Mapping[str, Any], *, max_drawdown_pct: float) -> str:
    max_dd = _float(row.get("selected_max_dd"))
    relative_return = _float(row.get("selected_relative_return"))
    if max_dd >= max_drawdown_pct:
        return "drawdown_control"
    if relative_return < TARGET_RELATIVE_RETURN_PP or not _bool(row.get("selected_target_gate")):
        return "trend_underparticipation"
    return "selection_gap"


def _fold_blockers(
    *,
    strict_gate_summary: Mapping[str, Any],
    checkpoint_selection_by_fold: pd.DataFrame,
    checkpoint_candidate_metrics: pd.DataFrame,
) -> list[dict[str, Any]]:
    max_drawdown_pct = _float(strict_gate_summary.get("max_walkforward_drawdown_pct"), 20.0)
    rows_by_fold = _selection_row_by_fold(checkpoint_selection_by_fold)
    blockers: list[dict[str, Any]] = []
    for fold in _required_folds(strict_gate_summary):
        row = rows_by_fold.get(fold)
        if row is None:
            blockers.append(
                {
                    "fold": fold,
                    "blocker": "missing_walkforward_fold",
                    "selected_candidate": "",
                    "selected_relative_return_pp": None,
                    "selected_max_dd_pct": None,
                }
            )
            continue
        candidate_gate = _bool(row.get("selected_candidate_gate"))
        target_gate = _bool(row.get("selected_target_gate"))
        drift_gate = _bool(row.get("selected_drift_gate"))
        max_dd = _float(row.get("selected_max_dd"))
        if candidate_gate and target_gate and drift_gate and max_dd < max_drawdown_pct:
            continue
        relative = _float(row.get("selected_relative_return"))
        ppo_drift = _ppo_drift_for_fold(checkpoint_candidate_metrics, fold, relative)
        blockers.append(
            {
                "fold": fold,
                "blocker": _blocker_for_row(row, max_drawdown_pct=max_drawdown_pct),
                "selected_candidate": str(row.get("selected_candidate", "")),
                "selected_kind": str(row.get("selected_kind", "")),
                "selected_relative_return_pp": round(relative, 6),
                "selected_max_dd_pct": round(max_dd, 6),
                "selected_target_gate": target_gate,
                "selected_drift_gate": drift_gate,
                "selected_candidate_gate": candidate_gate,
                **ppo_drift,
            }
        )
    return blockers


def _objective_terms(blockers: list[Mapping[str, Any]]) -> list[str]:
    blockers_seen = {str(row.get("blocker", "")) for row in blockers}
    terms = [
        "soxlbacktest_reference_alignment",
        "walkforward_relative_return_gap_penalty",
        "checkpoint_objective_loss",
    ]
    if "trend_underparticipation" in blockers_seen:
        terms.append("wf2020_trend_participation_floor")
        terms.append("drawdown_drift_penalty")
    if "drawdown_control" in blockers_seen:
        terms.append("wf_drawdown_control_state_cap")
        terms.append("drawdown_drift_penalty")
    return list(dict.fromkeys(terms))


def _decision(blockers: list[Mapping[str, Any]]) -> str:
    blockers_seen = {str(row.get("blocker", "")) for row in blockers}
    if "trend_underparticipation" in blockers_seen:
        return "train_bull_trend_participation_with_drawdown_drift_stop_no_promotion"
    if "drawdown_control" in blockers_seen:
        return "train_drawdown_control_objective_no_promotion"
    return "train_strict_retention_multi_objective_no_promotion"


def _stop_conditions(*, ppo_allowed_next: bool) -> list[str]:
    conditions = [
        "do_not_promote_missing_walkforward_windows",
        "do_not_promote_worst_wf_maxdd_ge_20",
        "do_not_promote_ytd_return_below_best_alignment",
        "do_not_promote_without_soxlbacktest_reference_alignment",
    ]
    if not ppo_allowed_next:
        conditions.append("do_not_continue_same_ppo_when_maxdd_drift_exceeds_2pp")
    return conditions


def derive_strict_retention_handoff(
    *,
    strict_gate_summary: Mapping[str, Any],
    checkpoint_selection_by_fold: pd.DataFrame,
    checkpoint_candidate_metrics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Derive the next no-promotion training handoff from strict retention evidence."""

    checkpoint_candidate_metrics = checkpoint_candidate_metrics if checkpoint_candidate_metrics is not None else pd.DataFrame()
    blockers = _fold_blockers(
        strict_gate_summary=strict_gate_summary,
        checkpoint_selection_by_fold=checkpoint_selection_by_fold,
        checkpoint_candidate_metrics=checkpoint_candidate_metrics,
    )
    strict_passing = int(strict_gate_summary.get("passing_profile_count") or 0)
    if strict_passing > 0 and not blockers:
        return _json_safe(
            {
                "decision": "candidate_promotion_review",
                "promotion_allowed": True,
                "actor_retention_allowed": True,
                "ppo_allowed_next": False,
                "diagnostic_training_allowed_next": False,
                "reference_baseline": strict_gate_summary.get("reference_profile"),
                "reference_return_pct": strict_gate_summary.get("reference_return_pct"),
                "required_best_return_pct": strict_gate_summary.get("required_best_return_pct"),
                "max_walkforward_drawdown_pct": strict_gate_summary.get("max_walkforward_drawdown_pct"),
                "required_wf_windows": _required_folds(strict_gate_summary),
                "fold_blockers": [],
                "objective_terms": [],
                "stop_conditions": [],
            }
        )

    ppo_allowed_next = not any(bool(row.get("ppo_drift_stop")) for row in blockers)
    primary_failed_fold = str(blockers[0].get("fold", "")) if blockers else ""
    return _json_safe(
        {
            "decision": _decision(blockers),
            "promotion_allowed": False,
            "actor_retention_allowed": False,
            "ppo_allowed_next": ppo_allowed_next,
            "diagnostic_training_allowed_next": True,
            "reference_baseline": strict_gate_summary.get("reference_profile"),
            "reference_return_pct": strict_gate_summary.get("reference_return_pct"),
            "reference_max_drawdown_pct": strict_gate_summary.get("reference_max_drawdown_pct"),
            "current_best_return_pct": strict_gate_summary.get("current_best_return_pct"),
            "required_best_return_pct": strict_gate_summary.get("required_best_return_pct"),
            "max_walkforward_drawdown_pct": strict_gate_summary.get("max_walkforward_drawdown_pct"),
            "required_wf_windows": _required_folds(strict_gate_summary),
            "primary_failed_fold": primary_failed_fold,
            "failed_fold_count": len(blockers),
            "fold_blockers": blockers,
            "objective_terms": _objective_terms(blockers),
            "stop_conditions": _stop_conditions(ppo_allowed_next=ppo_allowed_next),
        }
    )


def render_strict_retention_handoff(handoff: Mapping[str, Any]) -> str:
    lines = [
        "# Strict Retention Training Handoff",
        "",
        f"- decision: `{handoff.get('decision', '')}`",
        f"- promotion: `{'allowed' if handoff.get('promotion_allowed') else 'no promotion'}`",
        f"- actor_retention_allowed: `{handoff.get('actor_retention_allowed')}`",
        f"- ppo_allowed_next: `{handoff.get('ppo_allowed_next')}`",
        f"- reference_baseline: `{handoff.get('reference_baseline', '')}`",
        "",
        "## Objective Terms",
        "",
    ]
    for term in handoff.get("objective_terms", []):
        lines.append(f"- `{term}`")
    blockers = handoff.get("fold_blockers", [])
    if isinstance(blockers, list) and blockers:
        lines.extend(["", "## Fold Blockers", ""])
        for blocker in blockers:
            if not isinstance(blocker, Mapping):
                continue
            lines.append(
                "- "
                f"{blocker.get('fold')}: `{blocker.get('blocker')}`, "
                f"relative={blocker.get('selected_relative_return_pp')}pp, "
                f"MaxDD={blocker.get('selected_max_dd_pct')}%"
            )
    stop_conditions = handoff.get("stop_conditions", [])
    if isinstance(stop_conditions, list) and stop_conditions:
        lines.extend(["", "## Stop Conditions", ""])
        for condition in stop_conditions:
            lines.append(f"- `{condition}`")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "derive_strict_retention_handoff",
    "render_strict_retention_handoff",
]
