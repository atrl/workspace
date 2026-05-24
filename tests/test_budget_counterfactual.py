from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.compose_budget_usage_counterfactual import relative_output_label
from src.soxl_quant.app.budget_counterfactual import (
    BullParticipationGuardConfig,
    EquityDrawdownControlConfig,
    RecoveryReentryGuardConfig,
    build_budget_usage_counterfactual,
    render_budget_usage_review,
)
from src.soxl_quant.policy.two_speed_budget import TwoSpeedBudgetConfig


def test_build_budget_usage_counterfactual_scores_all_paths_and_keeps_audit_columns():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    slow = pd.DataFrame({"datetime": dates, "position": [0.20, 0.20, 0.70, 0.70, 0.32]})
    fast = pd.DataFrame({"datetime": dates, "position": [1.00, 0.00, 1.00, 0.00, 0.50]})
    price = pd.Series([100.0, 110.0, 99.0, 118.8, 112.86], index=dates, name="price")

    result = build_budget_usage_counterfactual(
        slow,
        fast,
        price,
        slow_label="slow_path",
        fast_label="fast_path",
        composed_label="composed_path",
        description="unit diagnostic",
        config=TwoSpeedBudgetConfig(trade_threshold=0.0, trade_lock_days=0),
        windows={"early": ("2020-01-01", "2020-01-03")},
    )

    assert result.daily["mode"].tolist() == [
        "flat_wait",
        "flat_wait",
        "trend_hold",
        "trend_hold",
        "defensive_hold",
    ]
    assert result.daily["position"].round(4).tolist() == [0.25, 0.0, 0.95, 0.45, 0.275]
    assert result.daily["slow_position"].tolist() == slow["position"].tolist()
    assert result.daily["fast_usage"].tolist() == fast["position"].tolist()

    assert set(result.payload["variants"]) == {"slow_path", "fast_path", "composed_path"}
    assert result.payload["variants"]["composed_path"]["slices"]["early"]["rows"] == 3
    assert (
        result.payload["comparison"]["composed_vs_fast"]["test_proxy"]["delta_return"]
        == result.payload["variants"]["composed_path"]["test_proxy"]["return"]
        - result.payload["variants"]["fast_path"]["test_proxy"]["return"]
    )


def test_bull_participation_guard_floors_strong_trend_to_fast_path():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    slow = pd.DataFrame({"datetime": dates, "position": [0.20, 0.20, 0.20, 0.20, 0.20]})
    fast = pd.DataFrame({"datetime": dates, "position": [0.70, 0.70, 0.70, 0.70, 0.70]})
    price = pd.Series([100.0, 103.0, 106.0, 109.0, 112.0], index=dates, name="price")

    result = build_budget_usage_counterfactual(
        slow,
        fast,
        price,
        slow_label="slow_path",
        fast_label="fast_path",
        composed_label="composed_path",
        description="unit diagnostic",
        config=TwoSpeedBudgetConfig(trade_threshold=0.0, trade_lock_days=0),
        bull_guard=BullParticipationGuardConfig(
            enabled=True,
            ma_window=2,
            momentum_window=1,
            high_window=2,
            min_periods=1,
            min_momentum=0.01,
            max_drawdown=0.02,
        ),
    )

    assert result.daily["bull_participation_guard"].tolist() == [False, True, True, True, True]
    assert result.daily["base_two_speed_position"].round(4).tolist() == [0.175] * 5
    assert result.daily["position"].round(4).tolist() == [0.175, 0.7, 0.7, 0.7, 0.7]


def test_recovery_reentry_guard_is_audited_in_counterfactual_payload():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    slow = pd.DataFrame({"datetime": dates, "position": [0.0] * 5})
    fast = pd.DataFrame({"datetime": dates, "position": [0.0] * 5})
    price = pd.Series([100.0, 101.0, 104.0, 108.0, 112.0], index=dates, name="price")
    features = pd.DataFrame(
        {
            "datetime": dates,
            "mom_10": [0.0, 0.10, 0.12, 0.13, 0.14],
            "mom_21": [0.0, 0.13, 0.15, 0.16, 0.17],
            "relative_strength_21": [0.0, 1.1, 1.2, 1.3, 1.4],
            "vol_of_vol_21": [0.0, 0.12, 0.13, 0.14, 0.15],
            "vix_percentile_252": [0.0, 0.70, 0.72, 0.74, 0.76],
            "below_anchor_ratio_63": [0.0, 0.10, 0.10, 0.10, 0.10],
            "anchor_gap_180": [0.0, 0.80, 0.80, 0.80, 0.80],
        }
    )

    result = build_budget_usage_counterfactual(
        slow,
        fast,
        price,
        slow_label="slow_path",
        fast_label="fast_path",
        composed_label="composed_path",
        description="unit diagnostic",
        config=TwoSpeedBudgetConfig(trade_threshold=0.0, trade_lock_days=0),
        feature_frame=features,
        recovery_guard=RecoveryReentryGuardConfig(
            enabled=True,
            min_target=0.35,
            max_target=0.45,
            max_base_target=0.20,
            min_mom_10=0.08,
            min_mom_21=0.12,
            min_relative_strength_21=1.0,
            min_vol_of_vol_21=0.10,
            max_vix_percentile_252=0.85,
            max_below_anchor_ratio_63=0.20,
            min_anchor_gap_180=0.50,
        ),
    )

    assert result.payload["config"]["recovery_reentry_guard"]["enabled"] is True
    assert result.payload["config"]["recovery_reentry_guard"]["action"] == "floor_low_target_in_high_vol_rebound"
    assert result.payload["config"]["recovery_reentry_guard"]["max_below_anchor_ratio_63"] == 0.20
    assert result.payload["config"]["recovery_reentry_guard"]["min_anchor_gap_180"] == 0.50
    assert result.daily["recovery_reentry_guard"].tolist() == [False, True, True, True, True]
    assert result.daily["position"].round(4).tolist() == [0.0, 0.35, 0.35, 0.35, 0.35]


def test_equity_drawdown_control_is_audited_in_counterfactual_payload():
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    slow = pd.DataFrame({"datetime": dates, "position": [0.70] * 4})
    fast = pd.DataFrame({"datetime": dates, "position": [1.00] * 4})
    price = pd.Series([100.0, 80.0, 84.0, 88.2], index=dates, name="price")

    result = build_budget_usage_counterfactual(
        slow,
        fast,
        price,
        slow_label="slow_path",
        fast_label="fast_path",
        composed_label="composed_path",
        description="unit diagnostic",
        config=TwoSpeedBudgetConfig(trade_threshold=0.0, trade_lock_days=5),
        drawdown_control=EquityDrawdownControlConfig(
            enabled=True,
            start_drawdown=0.10,
            full_drawdown=0.15,
            min_target=0.20,
            sell_trade_threshold=0.0,
            sell_trade_lock_days=0,
        ),
    )

    control = result.payload["config"]["equity_drawdown_control"]
    assert control["enabled"] is True
    assert control["start_drawdown"] == 0.10
    assert control["full_drawdown"] == 0.15
    assert control["min_target"] == 0.20
    assert control["action"] == "cap_target_after_strategy_equity_drawdown"
    assert result.daily["equity_drawdown_control"].tolist() == [False, True, True, True]
    assert result.daily["position"].round(4).tolist() == [0.95, 0.20, 0.20, 0.20]


def test_render_budget_usage_review_includes_governance_labels_and_deltas():
    payload = {
        "profile": "demo",
        "description": "unit diagnostic",
        "slow_label": "slow",
        "fast_label": "fast",
        "composed_label": "composed",
        "variants": {
            "slow": {
                "test_proxy": {"return": 1.0, "sharpe": 0.1, "max_dd": 2.0, "avg_pos": 0.3, "trades": 4},
                "slices": {
                    "test_slice": {"return": 1.0, "sharpe": 0.1, "max_dd": 2.0, "avg_pos": 0.3, "trades": 4}
                },
            },
            "fast": {
                "test_proxy": {"return": 2.0, "sharpe": 0.2, "max_dd": 3.0, "avg_pos": 0.4, "trades": 5},
                "slices": {
                    "test_slice": {"return": 2.0, "sharpe": 0.2, "max_dd": 3.0, "avg_pos": 0.4, "trades": 5}
                },
            },
            "composed": {
                "test_proxy": {"return": 3.0, "sharpe": 0.3, "max_dd": 4.0, "avg_pos": 0.5, "trades": 6},
                "slices": {
                    "test_slice": {"return": 3.0, "sharpe": 0.3, "max_dd": 4.0, "avg_pos": 0.5, "trades": 6}
                },
            },
        },
        "comparison": {"composed_vs_fast": {"test_proxy": {"delta_return": 1.0, "delta_max_dd": 1.0}}},
    }

    review = render_budget_usage_review(payload)

    assert "# demo" in review
    assert "| composed | 3.00% | 0.300 | 4.00% | 0.500 | 6 |" in review
    assert "| test_slice | 1.00% / 0.100 / 2.00% / pos 0.300 / trades 4 |" in review
    assert "composed_vs_fast" in review


def test_relative_output_label_accepts_relative_and_absolute_paths():
    rel = relative_output_label(Path("runtime/model/card/demo"))
    abs_path = relative_output_label(Path.cwd() / "runtime/model/card/demo")

    assert rel == "runtime/model/card/demo"
    assert abs_path == "runtime/model/card/demo"
