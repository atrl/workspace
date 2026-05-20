from __future__ import annotations

import pandas as pd

from scripts.evaluate_selective_release_walkforward import (
    eligible_decision_trace_rows,
    execution_primary_blocker,
    gate_eligible_mask,
    variant_subset,
    walk_forward_actor_specs,
)


def test_walk_forward_actor_specs_follow_profile_artifact_layout():
    specs = walk_forward_actor_specs("balanced_rule_residual_v22")

    assert [spec.window for spec in specs] == ["wf_2020", "wf_2022", "wf_2023_2024"]
    assert specs[0].actor_path.as_posix().endswith(
        "runtime/model/card/balanced_rule_residual_v22_wf_2020/"
        "soxl_balanced_rule_residual_v22_wf_2020_actor.pt"
    )
    assert specs[2].start == "2024-01-01"
    assert specs[2].end == "2024-12-31"


def test_retained_variant_subset_keeps_baseline_and_recovery_gate():
    variants = variant_subset(retained_only=True)

    assert [name for name, _ in variants] == ["baseline", "selective_release_recovery_v1"]


def test_gate_eligible_mask_counts_signal_confirmation_inside_regime_band():
    frame = pd.DataFrame(
        {
            "regime_risk": [0.50, 0.70, 0.50],
            "RB_COOLDOWN_PROXY": [0.30, 0.30, 0.10],
            "TRANS_VIX_DIVERGENCE": [0.01, 0.10, 0.10],
            "CICC_RETURN_VOLUME_SYNC": [0.72, 0.72, 0.72],
        }
    )
    config = {
        "max_regime_risk": 0.60,
        "cooldown_threshold": 0.20,
        "vix_threshold": 0.05,
        "volume_sync_threshold": 0.70,
    }

    assert gate_eligible_mask(frame, config).tolist() == [True, False, False]


def test_eligible_decision_trace_rows_keep_gate_ready_execution_state():
    frame = pd.DataFrame(
        {
            "regime_risk": [0.50, 0.50],
            "RB_COOLDOWN_PROXY": [0.30, 0.10],
            "TRANS_VIX_DIVERGENCE": [0.01, 0.10],
            "CICC_RETURN_VOLUME_SYNC": [0.72, 0.72],
        },
        index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
    )
    config = {
        "max_regime_risk": 0.60,
        "cooldown_threshold": 0.20,
        "vix_threshold": 0.05,
        "volume_sync_threshold": 0.70,
    }
    trace = [
        {
            "date": "2024-01-03",
            "current_pos": 0.61,
            "target_before_overlay": 0.60,
            "target_after_overlay": 0.65,
            "target_after_guards": 0.61,
            "min_rebalance": 0.0,
            "trade_lock_before": 1,
            "buy_cooldown_before": 0,
            "blocked_by_trade_lock": True,
            "blocked_by_buy_cooldown": False,
            "blocked_by_lot_rounding": False,
            "order_count": 0,
        },
        {
            "date": "2024-01-04",
            "current_pos": 0.50,
            "target_before_overlay": 0.50,
            "target_after_overlay": 0.50,
            "target_after_guards": 0.50,
            "min_rebalance": 0.04,
            "trade_lock_before": 0,
            "buy_cooldown_before": 0,
            "blocked_by_trade_lock": False,
            "blocked_by_buy_cooldown": False,
            "blocked_by_lot_rounding": False,
            "order_count": 0,
        },
    ]

    rows = eligible_decision_trace_rows(
        trace,
        frame,
        config,
        actor_window="wf_2023_2024",
        variant="selective_release_recovery_target_lift_v1",
    )

    assert rows == [
        {
            "actor_window": "wf_2023_2024",
            "variant": "selective_release_recovery_target_lift_v1",
            "date": "2024-01-03",
            "current_pos": 0.61,
            "target_before_overlay": 0.60,
            "target_after_overlay": 0.65,
            "target_after_guards": 0.61,
            "min_rebalance": 0.0,
            "trade_lock_before": 1,
            "buy_cooldown_before": 0,
            "blocked_by_trade_lock": True,
            "blocked_by_buy_cooldown": False,
            "blocked_by_lot_rounding": False,
            "order_count": 0,
            "primary_blocker": "trade_lock",
        }
    ]


def test_execution_primary_blocker_prioritizes_upstream_guards():
    row = {
        "blocked_by_trade_lock": True,
        "blocked_by_buy_cooldown": False,
        "blocked_by_min_rebalance": True,
        "blocked_by_step_lock": False,
        "blocked_by_lot_rounding": True,
        "order_count": 0,
    }

    assert execution_primary_blocker(row) == "trade_lock"
    assert execution_primary_blocker({**row, "blocked_by_trade_lock": False}) == "min_rebalance"
    assert execution_primary_blocker({**row, "blocked_by_trade_lock": False, "order_count": 1}) == "ordered"
