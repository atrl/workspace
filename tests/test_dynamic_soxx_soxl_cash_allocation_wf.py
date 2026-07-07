from __future__ import annotations

import pandas as pd

from src.workflow.dynamic_soxx_soxl_cash_allocation_wf import (
    _annual_goal_summary,
    _dynamic_allocation_candidate_configs,
    _select_dynamic_allocation_candidate,
    _simulate_dynamic_allocation_policy,
)


def test_dynamic_allocation_policy_sums_to_one_and_uses_cash_under_tail_risk():
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "instrument": ["SOXL", "SOXL"],
            "etf_open_to_open_return": [0.06, -0.05],
            "underlying_open_to_open_return": [0.02, -0.02],
        }
    )
    predictions = pd.DataFrame(
        {
            "relative_alpha_probability": [0.90, 0.20],
            "underlying_positive_probability": [0.85, 0.10],
            "tail_risk_probability": [0.05, 0.95],
        }
    )

    candidate = _dynamic_allocation_candidate_configs()[0]
    metrics, positions = _simulate_dynamic_allocation_policy(
        frame,
        predictions,
        policy=candidate,
        cost_per_turnover=0.0,
    )

    weight_sum = positions["weight_soxx"] + positions["weight_soxl"] + positions["weight_cash"]
    assert weight_sum.round(10).eq(1.0).all()
    assert positions.loc[0, "weight_soxl"] > 0.25
    assert positions.loc[0, "weight_cash"] < 0.25
    assert positions.loc[1, "weight_cash"] >= 0.80
    assert positions.loc[1, "weight_soxl"] <= 0.05
    assert metrics["max_soxl_weight"] <= candidate["max_soxl_weight"]


def test_dynamic_selector_prefers_candidate_with_proxy_generalization():
    metrics = pd.DataFrame(
        [
            _metric_row("soxl_only", "SOXL", 2025, excess=35.0, maxdd=-6.0),
            _metric_row("soxl_only", "TQQQ", 2025, excess=-12.0, maxdd=-18.0),
            _metric_row("soxl_only", "UPRO", 2025, excess=-8.0, maxdd=-16.0),
            _metric_row("generalized", "SOXL", 2025, excess=24.0, maxdd=-7.0),
            _metric_row("generalized", "TQQQ", 2025, excess=4.0, maxdd=-12.0),
            _metric_row("generalized", "UPRO", 2025, excess=3.0, maxdd=-11.0),
        ]
    )

    selected = _select_dynamic_allocation_candidate(
        metrics,
        outer_test_year=2026,
        required_proxy_assets=("TQQQ", "UPRO"),
    )

    assert selected["candidate"] == "generalized"
    assert selected["generalization_selection_satisfied"] is True
    assert selected["selection_gate_status"] == "pass"
    assert selected["proxy_generalization_gate_applied"] is True
    assert selected["selection_uses_outer_test"] is False


def test_dynamic_annual_goal_uses_soxx_excess_20_points_and_ten_drawdown_floor():
    annual = pd.DataFrame(
        [
            {"outer_test_year": 2023, "test_excess_vs_soxx_pct": 21.0, "test_max_drawdown_pct": -9.5},
            {"outer_test_year": 2024, "test_excess_vs_soxx_pct": 19.0, "test_max_drawdown_pct": -8.0},
            {"outer_test_year": 2025, "test_excess_vs_soxx_pct": 30.0, "test_max_drawdown_pct": -12.0},
        ]
    )

    goal = _annual_goal_summary(annual, required_years=(2023, 2024, 2025), excess_return_floor_pct=20.0, max_drawdown_floor_pct=-10.0)

    assert goal["target_pass_test_years"] == [2023]
    assert goal["required_policy_satisfied"] is False
    assert "annual_excess_or_drawdown_gate_failed" in goal["blocking_reasons"]
    assert goal["return_gate"] == "test_return_pct_ge_soxx_buyhold_return_pct_plus_20pp"
    assert goal["max_drawdown_floor_pct"] == -10.0


def _metric_row(candidate: str, asset: str, year: int, *, excess: float, maxdd: float) -> dict:
    return {
        "candidate": candidate,
        "model": "lgbm",
        "policy_id": "balanced",
        "feature_set": "base",
        "inner_year": year,
        "heldout_asset": asset,
        "selection_return_pct": 30.0 + excess,
        "selection_max_drawdown_pct": maxdd,
        "selection_total_turnover": 4.0,
        "selection_avg_equity_weight": 0.70,
        "selection_avg_soxl_weight": 0.20,
        "benchmark_return_pct": 30.0,
        "excess_vs_underlying_pct": excess,
        "passes_excess_gate": excess >= 20.0 if asset == "SOXL" else excess >= 0.0,
        "passes_drawdown_gate": maxdd >= -10.0 if asset == "SOXL" else maxdd >= -20.0,
        "selection_uses_outer_test": False,
    }
