from __future__ import annotations

import pandas as pd

from src.workflow.intraday_execution_sleeve_v2 import (
    INTRADAY_EXECUTION_SLEEVE_V2_EXPERIMENT,
    TRIPLE_RISK_GATE_COL,
    TREND_PROTECTION_FLOOR_COL,
    _annual_goal_summary,
    _attach_execution_sleeve_context,
    _select_validation_sleeve_policy,
    _select_validation_regime_template,
)


def test_execution_sleeve_context_requires_daily_intraday_and_breadth_risk():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "probability": [0.72, 0.18, 0.22],
            "target_position": [0.80, 0.70, 0.60],
            "intraday15m_soxl_late_ret_lag1": [-0.050, -0.060, -0.010],
            "intraday15m_soxl_minus_3x_soxx_ret_lag1": [-0.010, -0.008, -0.002],
            "soxx_const_breadth_ret5_pos_lag1": [0.70, 0.32, 0.25],
            "soxx_const_downside_ret5_lt_m5_lag1": [0.05, 0.35, 0.40],
            "soxx_ret_20": [0.08, -0.04, 0.10],
            "trend_efficiency_20": [0.65, 0.20, 0.70],
        }
    )

    out = _attach_execution_sleeve_context(frame)

    assert out[TRIPLE_RISK_GATE_COL].tolist() == [0.0, 1.0, 0.0]
    assert out[TREND_PROTECTION_FLOOR_COL].tolist() == [1.0, 0.0, 1.0]


def test_select_validation_sleeve_policy_prefers_retention_and_buyhold_goal():
    candidates = pd.DataFrame(
        [
            {
                "policy_id": "aggressive_dd_fail",
                "cost_bps": 20.0,
                "validation_return_retention_pct": 120.0,
                "validation_max_drawdown_pct": -12.0,
                "validation_passes_target": True,
                "test_return_pct": 260.0,
                "test_max_drawdown_pct": -24.0,
                "buyhold_return_pct": 200.0,
                "test_year": 2026,
            },
            {
                "policy_id": "balanced_goal_pass",
                "cost_bps": 20.0,
                "validation_return_retention_pct": 91.0,
                "validation_max_drawdown_pct": -11.0,
                "validation_passes_target": True,
                "test_return_pct": 230.0,
                "test_max_drawdown_pct": -16.0,
                "buyhold_return_pct": 200.0,
                "test_year": 2026,
            },
            {
                "policy_id": "low_retention",
                "cost_bps": 20.0,
                "validation_return_retention_pct": 55.0,
                "validation_max_drawdown_pct": -5.0,
                "validation_passes_target": True,
                "test_return_pct": 250.0,
                "test_max_drawdown_pct": -10.0,
                "buyhold_return_pct": 200.0,
                "test_year": 2026,
            },
        ]
    )

    selected = _select_validation_sleeve_policy(candidates, required_cost_bps=(20.0,))

    assert selected["policy_id"] == "balanced_goal_pass"
    assert selected["selection_uses_test_segment"] is False
    assert selected["return_retention_floor_pct"] == 85.0
    assert selected["annual_goal_gate"] == "report_only_not_selection"


def test_annual_goal_summary_requires_all_years_to_beat_buyhold_and_drawdown_floor():
    walk_forward = pd.DataFrame(
        [
            {
                "test_year": 2023,
                "test_return_pct": 175.0,
                "test_max_drawdown_pct": -19.0,
                "buyhold_return_pct": 169.0,
            },
            {
                "test_year": 2024,
                "test_return_pct": 20.0,
                "test_max_drawdown_pct": -18.0,
                "buyhold_return_pct": 6.5,
            },
            {
                "test_year": 2025,
                "test_return_pct": 75.0,
                "test_max_drawdown_pct": -15.0,
                "buyhold_return_pct": 77.0,
            },
            {
                "test_year": 2026,
                "test_return_pct": 360.0,
                "test_max_drawdown_pct": -19.5,
                "buyhold_return_pct": 335.0,
            },
        ]
    )

    summary = _annual_goal_summary(walk_forward)

    assert summary["experiment"] == INTRADAY_EXECUTION_SLEEVE_V2_EXPERIMENT
    assert summary["target_pass_test_years"] == [2023, 2024, 2026]
    assert summary["required_policy_satisfied"] is False
    assert "annual_target_gate_failed" in summary["blocking_reasons"]


def test_select_validation_regime_template_uses_validation_regime_not_test_results():
    assert (
        _select_validation_regime_template({"return_pct": -86.0, "max_drawdown_pct": -90.0})["template_id"]
        == "post_bear_ret60_late_confirm"
    )
    assert (
        _select_validation_regime_template({"return_pct": 169.0, "max_drawdown_pct": -49.0})["template_id"]
        == "post_rebound_low_ret60_defense"
    )
    assert (
        _select_validation_regime_template({"return_pct": 6.0, "max_drawdown_pct": -66.0})["template_id"]
        == "range_repair_high_trend_efficiency"
    )
    selected = _select_validation_regime_template({"return_pct": 77.0, "max_drawdown_pct": -77.0})
    assert selected["template_id"] == "source_daily_lgb_xgb_ml"
    assert selected["selection_uses_test_segment"] is False
    assert selected["selection_basis"] == "validation_regime_only"
