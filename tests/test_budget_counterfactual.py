from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.compose_budget_usage_counterfactual import relative_output_label
from src.soxl_quant.app.budget_counterfactual import (
    BullParticipationGuardConfig,
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
