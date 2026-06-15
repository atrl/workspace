from __future__ import annotations

import json

from src.soxl_quant.app.bc_warmstart_strict_gate import (
    derive_bc_warmstart_strict_gate,
    render_bc_warmstart_strict_gate,
)


def _strict_candidate_filter() -> dict[str, object]:
    return {
        "decision": "strict_candidate_retention_filter_no_promotable_candidates",
        "reference_baseline": "soxl_backtest_policy_reimplementation_v1",
        "passing_profile_count": 0,
        "promotable_profile_count": 0,
        "actor_retention_allowed_count": 0,
        "reference_visible_count": 2,
        "ppo_allowed_next": False,
        "next_mechanism": "train_bull_trend_participation_with_drawdown_drift_stop_no_promotion",
        "max_walkforward_drawdown_pct": 20.0,
        "required_best_return_pct": 213.143462,
        "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
    }


def _target_contract() -> dict[str, object]:
    return {
        "experiment": "bull_reentry_joint_target_contract_v1",
        "decision": "joint_target_contract_bull_reentry_gate_ready_for_training_handoff",
        "training_allowed_next": True,
        "promotion_allowed": False,
        "actor_retention_allowed": False,
        "rows": 3,
        "bull_reentry_target_rows": 2,
        "eval_overlap_rows": 0,
        "risk_conflict_target_rows": 0,
        "next_mechanism": "wire_fold_local_bull_reentry_joint_targets_into_bc_warm_start_then_full_walkforward_gate",
        "required_retention_gates": {
            "required_wf_windows": ["wf_2020", "wf_2022", "wf_2023_2024"],
            "max_walkforward_drawdown_pct": 20.0,
            "reference_baseline": "soxl_backtest_policy_reimplementation_v1",
            "required_best_return_pct": 213.143462,
        },
    }


def test_gate_allows_bc_warmstart_training_but_blocks_promotion_and_retention():
    gate = derive_bc_warmstart_strict_gate(
        strict_candidate_filter=_strict_candidate_filter(),
        target_contract=_target_contract(),
    )

    assert gate["decision"] == "bull_reentry_bc_warmstart_allowed_no_promotion"
    assert gate["bc_warm_start_allowed"] is True
    assert gate["training_allowed_next"] is True
    assert gate["promotion_allowed"] is False
    assert gate["actor_retention_allowed"] is False
    assert gate["ppo_allowed_next"] is False
    assert gate["reference_baseline"] == "soxl_backtest_policy_reimplementation_v1"
    assert gate["required_gates"]["required_wf_windows"] == ["wf_2020", "wf_2022", "wf_2023_2024"]
    assert gate["required_gates"]["max_walkforward_drawdown_pct"] == 20.0
    assert gate["required_gates"]["required_best_return_pct"] == 213.143462
    assert gate["source_target_contract"]["bull_reentry_target_rows"] == 2
    assert gate["source_target_contract"]["eval_overlap_rows"] == 0
    assert gate["strict_candidate_filter"]["actor_retention_allowed_count"] == 0
    assert "strict_candidate_retention_filter_no_promotable_candidates" in gate["blocked_promotions"]
    assert gate["required_next_gate"] == "full_fixed_split_and_walkforward_retention_gate"
    assert gate["next_mechanism"] == "wire_fold_local_bull_reentry_joint_targets_into_bc_warm_start_then_full_walkforward_gate"
    json.dumps(gate)


def test_gate_blocks_bc_warmstart_when_target_contract_overlaps_eval_or_conflicts_with_risk():
    contract = _target_contract()
    contract["eval_overlap_rows"] = 1
    contract["risk_conflict_target_rows"] = 2

    gate = derive_bc_warmstart_strict_gate(
        strict_candidate_filter=_strict_candidate_filter(),
        target_contract=contract,
    )

    assert gate["decision"] == "bull_reentry_bc_warmstart_blocked_no_promotion"
    assert gate["bc_warm_start_allowed"] is False
    assert gate["training_allowed_next"] is False
    assert gate["promotion_allowed"] is False
    assert gate["actor_retention_allowed"] is False
    assert "target_contract_has_eval_overlap" in gate["blockers"]
    assert "target_contract_has_risk_conflicts" in gate["blockers"]


def test_gate_render_names_reference_baseline_and_required_walkforward_windows():
    gate = derive_bc_warmstart_strict_gate(
        strict_candidate_filter=_strict_candidate_filter(),
        target_contract=_target_contract(),
    )

    rendered = render_bc_warmstart_strict_gate(gate)

    assert "SOXLBacktest" in rendered
    assert "wf_2020" in rendered
    assert "wf_2022" in rendered
    assert "wf_2023_2024" in rendered
    assert "no promotion" in rendered
