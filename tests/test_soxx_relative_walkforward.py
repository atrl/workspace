from __future__ import annotations

import pandas as pd

from scripts.evaluate_soxx_relative_walkforward import (
    review_decision,
    soxx_eligible_decision_trace_rows,
    soxx_gate_eligible_mask,
)


def test_soxx_gate_eligible_mask_requires_weak_underlying_and_no_reversion_cushion():
    frame = pd.DataFrame(
        {
            "regime_risk": [0.62, 0.30, 0.62, 0.62],
            "SOXX_BIAS_10": [-0.04, -0.04, -0.01, -0.04],
            "PRICE_RATIO_MOM": [-0.02, -0.02, -0.02, -0.02],
            "SIGNED_TE_REVERSION_10": [0.20, 0.20, 0.20, 0.80],
        }
    )
    config = {
        "min_regime_risk": 0.45,
        "max_regime_risk": 0.90,
        "soxx_bias_max": -0.02,
        "price_ratio_mom_max": 0.0,
        "signed_te_reversion_max": 0.50,
    }

    assert soxx_gate_eligible_mask(frame, config).tolist() == [True, False, False, False]


def test_soxx_eligible_decision_trace_rows_tracks_cap_and_primary_blocker():
    frame = pd.DataFrame(
        {
            "regime_risk": [0.62, 0.62],
            "SOXX_BIAS_10": [-0.04, -0.01],
            "PRICE_RATIO_MOM": [-0.02, -0.02],
            "SIGNED_TE_REVERSION_10": [0.20, 0.20],
        },
        index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
    )
    config = {
        "min_regime_risk": 0.45,
        "max_regime_risk": 0.90,
        "soxx_bias_max": -0.02,
        "price_ratio_mom_max": 0.0,
        "signed_te_reversion_max": 0.50,
    }
    trace = [
        {
            "date": "2024-01-03",
            "target_before_overlay": 0.58,
            "target_after_overlay": 0.35,
            "blocked_by_trade_lock": False,
            "blocked_by_buy_cooldown": False,
            "blocked_by_min_rebalance": True,
            "blocked_by_step_lock": False,
            "blocked_by_lot_rounding": False,
            "order_count": 0,
        },
        {
            "date": "2024-01-04",
            "target_before_overlay": 0.58,
            "target_after_overlay": 0.58,
            "blocked_by_trade_lock": False,
            "blocked_by_buy_cooldown": False,
            "blocked_by_min_rebalance": False,
            "blocked_by_step_lock": False,
            "blocked_by_lot_rounding": False,
            "order_count": 0,
        },
    ]

    rows = soxx_eligible_decision_trace_rows(
        trace,
        frame,
        config,
        actor_window="wf_2023_2024",
        variant="soxx_relative_weakness_cap_v1",
    )

    assert rows == [
        {
            "actor_window": "wf_2023_2024",
            "variant": "soxx_relative_weakness_cap_v1",
            "date": "2024-01-03",
            "target_before_overlay": 0.58,
            "target_after_overlay": 0.35,
            "blocked_by_trade_lock": False,
            "blocked_by_buy_cooldown": False,
            "blocked_by_min_rebalance": True,
            "blocked_by_step_lock": False,
            "blocked_by_lot_rounding": False,
            "order_count": 0,
            "cap_changed": True,
            "primary_blocker": "min_rebalance",
        }
    ]


def test_review_decision_requires_walk_forward_position_change_not_only_eligibility():
    rows = [
        {
            "actor_window": "wf_2022",
            "variant": "baseline",
            "return": -30.0,
            "sharpe": -0.8,
            "max_dd": 40.0,
            "avg_pos": 0.33,
            "changed_days": 0,
            "eligible_days": 0,
            "cap_changed_days": 0,
        },
        {
            "actor_window": "wf_2023_2024",
            "variant": "baseline",
            "return": -20.0,
            "sharpe": -0.2,
            "max_dd": 38.0,
            "avg_pos": 0.42,
            "changed_days": 0,
            "eligible_days": 0,
            "cap_changed_days": 0,
        },
        {
            "actor_window": "wf_2022",
            "variant": "soxx_relative_weakness_cap_v1",
            "return": -30.0,
            "sharpe": -0.8,
            "max_dd": 40.0,
            "avg_pos": 0.33,
            "changed_days": 0,
            "eligible_days": 10,
            "cap_changed_days": 0,
        },
        {
            "actor_window": "wf_2023_2024",
            "variant": "soxx_relative_weakness_cap_v1",
            "return": -10.0,
            "sharpe": -0.1,
            "max_dd": 37.0,
            "avg_pos": 0.40,
            "changed_days": 0,
            "eligible_days": 8,
            "cap_changed_days": 0,
        },
    ]

    decision = review_decision(rows)

    assert decision["decision"] == "blocked"
    assert decision["checks"]["wf_2023_2024_recovery_improved"] is True
    assert decision["checks"]["walk_forward_position_changed"] is False
