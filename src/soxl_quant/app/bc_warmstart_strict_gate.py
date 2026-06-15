from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


REFERENCE_PROFILE = "soxl_backtest_policy_reimplementation_v1"
REQUIRED_NEXT_GATE = "full_fixed_split_and_walkforward_retention_gate"


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


def _summary(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = payload.get("summary")
    return summary if isinstance(summary, Mapping) else payload


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value) if value is not None else False


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _required_gates(
    *,
    strict_candidate_filter: Mapping[str, Any],
    target_contract: Mapping[str, Any],
) -> dict[str, Any]:
    contract_gates = target_contract.get("required_retention_gates")
    contract_gates = contract_gates if isinstance(contract_gates, Mapping) else {}
    return {
        "required_wf_windows": [
            str(item)
            for item in (
                contract_gates.get("required_wf_windows")
                or strict_candidate_filter.get("required_wf_windows")
                or []
            )
        ],
        "max_walkforward_drawdown_pct": contract_gates.get(
            "max_walkforward_drawdown_pct",
            strict_candidate_filter.get("max_walkforward_drawdown_pct"),
        ),
        "reference_baseline": contract_gates.get(
            "reference_baseline",
            strict_candidate_filter.get("reference_baseline") or REFERENCE_PROFILE,
        ),
        "required_best_return_pct": contract_gates.get(
            "required_best_return_pct",
            strict_candidate_filter.get("required_best_return_pct"),
        ),
    }


def _blockers(
    *,
    strict_candidate_filter: Mapping[str, Any],
    target_contract: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not _as_bool(target_contract.get("training_allowed_next")):
        blockers.append("target_contract_training_not_allowed")
    if _as_bool(target_contract.get("promotion_allowed")) or _as_bool(target_contract.get("actor_retention_allowed")):
        blockers.append("target_contract_attempts_promotion_or_retention")
    if _as_int(target_contract.get("bull_reentry_target_rows")) <= 0:
        blockers.append("target_contract_has_no_bull_reentry_rows")
    if _as_int(target_contract.get("eval_overlap_rows")) > 0:
        blockers.append("target_contract_has_eval_overlap")
    if _as_int(target_contract.get("risk_conflict_target_rows")) > 0:
        blockers.append("target_contract_has_risk_conflicts")
    if _as_int(strict_candidate_filter.get("passing_profile_count")) > 0:
        blockers.append("strict_filter_has_passing_profiles_use_promotion_review")
    if _as_int(strict_candidate_filter.get("actor_retention_allowed_count")) > 0:
        blockers.append("strict_filter_allows_actor_retention_use_promotion_review")
    return blockers


def derive_bc_warmstart_strict_gate(
    *,
    strict_candidate_filter: Mapping[str, Any],
    target_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose a conservative BC warm-start handoff from strict retention evidence.

    This helper is intentionally not a promotion gate. It only says whether a
    target-contract handoff is clean enough to start fold-local BC warm-start
    training before rerunning the full fixed-split and walk-forward gate.
    """

    strict_summary = _summary(strict_candidate_filter)
    blockers = _blockers(strict_candidate_filter=strict_summary, target_contract=target_contract)
    allowed = not blockers
    required_gates = _required_gates(
        strict_candidate_filter=strict_summary,
        target_contract=target_contract,
    )
    target_experiment = str(target_contract.get("experiment") or "")
    strict_decision = str(strict_summary.get("decision") or "")
    blocked_promotions = [
        item
        for item in [
            strict_decision,
            "no_actor_retention_allowed" if _as_int(strict_summary.get("actor_retention_allowed_count")) == 0 else "",
            "reference_baseline_comparison_only",
        ]
        if item
    ]
    return _json_safe(
        {
            "decision": (
                "bull_reentry_bc_warmstart_allowed_no_promotion"
                if allowed
                else "bull_reentry_bc_warmstart_blocked_no_promotion"
            ),
            "bc_warm_start_allowed": allowed,
            "training_allowed_next": allowed,
            "promotion_allowed": False,
            "actor_retention_allowed": False,
            "ppo_allowed_next": False,
            "reference_baseline": required_gates["reference_baseline"],
            "required_gates": required_gates,
            "required_next_gate": REQUIRED_NEXT_GATE,
            "next_mechanism": (
                target_contract.get("next_mechanism")
                or "wire_fold_local_targets_into_bc_warm_start_then_full_walkforward_gate"
            ),
            "source_target_contract": {
                "experiment": target_experiment,
                "decision": target_contract.get("decision"),
                "rows": target_contract.get("rows"),
                "bull_reentry_target_rows": target_contract.get("bull_reentry_target_rows"),
                "eval_overlap_rows": target_contract.get("eval_overlap_rows"),
                "risk_conflict_target_rows": target_contract.get("risk_conflict_target_rows"),
            },
            "strict_candidate_filter": {
                "decision": strict_summary.get("decision"),
                "passing_profile_count": strict_summary.get("passing_profile_count"),
                "promotable_profile_count": strict_summary.get("promotable_profile_count"),
                "actor_retention_allowed_count": strict_summary.get("actor_retention_allowed_count"),
                "ppo_allowed_next": strict_summary.get("ppo_allowed_next"),
                "next_mechanism": strict_summary.get("next_mechanism"),
            },
            "blocked_promotions": blocked_promotions,
            "blockers": blockers,
            "stop_conditions": [
                "do_not_promote_bc_warmstart_without_full_fixed_split_gate",
                "do_not_retain_actor_without_required_walkforward_windows",
                "do_not_continue_ppo_until_bc_candidate_passes_drawdown_drift_screen",
            ],
        }
    )


def _format_profile(profile: Any) -> str:
    return "SOXLBacktest" if str(profile) == REFERENCE_PROFILE else str(profile)


def render_bc_warmstart_strict_gate(gate: Mapping[str, Any]) -> str:
    required_gates = gate.get("required_gates") if isinstance(gate.get("required_gates"), Mapping) else {}
    lines = [
        "# BC Warm-start Strict Gate",
        "",
        f"- decision: `{gate.get('decision', '')}`",
        f"- training: `{'allowed' if gate.get('training_allowed_next') else 'blocked'}`",
        f"- promotion: `{'allowed' if gate.get('promotion_allowed') else 'no promotion'}`",
        f"- actor_retention_allowed: `{gate.get('actor_retention_allowed')}`",
        f"- reference_baseline: `{_format_profile(gate.get('reference_baseline', ''))}`",
        f"- required_next_gate: `{gate.get('required_next_gate', '')}`",
        "",
        "## Required Gates",
        "",
        f"- Max walk-forward drawdown: `{required_gates.get('max_walkforward_drawdown_pct')}`",
        f"- Required best return: `{required_gates.get('required_best_return_pct')}`",
        f"- Walk-forward windows: `{', '.join(str(item) for item in required_gates.get('required_wf_windows') or [])}`",
    ]
    blockers = gate.get("blockers")
    if isinstance(blockers, list) and blockers:
        lines.extend(["", "## Blockers", ""])
        for blocker in blockers:
            lines.append(f"- `{blocker}`")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "derive_bc_warmstart_strict_gate",
    "render_bc_warmstart_strict_gate",
]
