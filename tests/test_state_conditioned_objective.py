from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.state_conditioned_objective import (
    StateConditionedObjectiveConfig,
    daily_path_metrics,
    max_drawdown_interval_metrics,
    render_state_conditioned_objective_review,
    score_state_conditioned_objective,
    summarize_state_conditioned_objective,
)


def test_state_conditioned_objective_prefers_joint_walkforward_and_rebound_fit():
    metrics_by_candidate = {
        "high_return_bad_risk": {
            "wf_2020": {"relative_return": 100.0, "max_dd": 32.0, "avg_pos": 0.75},
            "wf_2022": {"relative_return": 80.0, "max_dd": 26.0, "avg_pos": 0.55},
        },
        "balanced_state_candidate": {
            "wf_2020": {"relative_return": 34.0, "max_dd": 14.0, "avg_pos": 0.42},
            "wf_2022": {"relative_return": 38.0, "max_dd": 13.0, "avg_pos": 0.22},
        },
        "safe_underparticipant": {
            "wf_2020": {"relative_return": 45.0, "max_dd": 11.0, "avg_pos": 0.18},
            "wf_2022": {"relative_return": 52.0, "max_dd": 10.0, "avg_pos": 0.09},
        },
    }
    cluster_metrics_by_candidate = {
        "high_return_bad_risk": {
            "C2026": {"relative_return": -130.0, "max_dd": 24.0, "avg_pos": 0.15}
        },
        "balanced_state_candidate": {
            "C2026": {"relative_return": -20.0, "max_dd": 9.0, "avg_pos": 0.36}
        },
        "safe_underparticipant": {
            "C2026": {"relative_return": -110.0, "max_dd": 5.0, "avg_pos": 0.08}
        },
    }

    scored = score_state_conditioned_objective(
        metrics_by_candidate,
        cluster_metrics_by_candidate=cluster_metrics_by_candidate,
    )

    assert scored.iloc[0]["candidate"] == "balanced_state_candidate"
    assert scored.iloc[0]["objective_pass"]
    assert scored.set_index("candidate").loc["high_return_bad_risk", "drawdown_loss"] > 0
    assert scored.set_index("candidate").loc["safe_underparticipant", "participation_loss"] > 0


def test_state_conditioned_objective_blocks_promotion_when_no_cluster_passes():
    metrics_by_candidate = {
        "closest": {
            "wf_2020": {"relative_return": 55.0, "max_dd": 26.0, "avg_pos": 0.28},
            "wf_2022": {"relative_return": 60.0, "max_dd": 18.0, "avg_pos": 0.18},
        }
    }
    cluster_metrics_by_candidate = {
        "closest": {"C2026": {"relative_return": -85.0, "max_dd": 7.0, "avg_pos": 0.31}}
    }

    scored = score_state_conditioned_objective(
        metrics_by_candidate,
        cluster_metrics_by_candidate=cluster_metrics_by_candidate,
    )
    summary = summarize_state_conditioned_objective(scored)

    assert summary["decision"] == "no_objective_pass_candidate"
    assert summary["next_mechanism"] == "train_state_conditioned_multi_objective"
    assert summary["selected_candidate"] == "closest"
    assert "walkforward_drawdown_gap_penalty" in summary["required_terms"]
    assert "high_vol_rebound_participation_floor" in summary["required_terms"]
    assert "no runtime promotion" in render_state_conditioned_objective_review(scored, summary)


def test_daily_path_metrics_computes_relative_return_drawdown_and_position():
    daily = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=5, freq="B"),
            "strategy_nav": [100.0, 110.0, 105.0, 120.0, 130.0],
            "buyhold_nav": [100.0, 120.0, 125.0, 130.0, 140.0],
            "position": [0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )

    metrics = daily_path_metrics(daily)

    assert round(metrics["return"], 2) == 30.0
    assert round(metrics["buy_and_hold_return"], 2) == 40.0
    assert round(metrics["relative_return"], 2) == -10.0
    assert round(metrics["max_dd"], 2) == 4.55
    assert metrics["avg_pos"] == 0.4


def test_daily_path_metrics_accepts_router_position_alias():
    daily = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=3, freq="B"),
            "strategy_nav": [100.0, 105.0, 110.0],
            "buyhold_nav": [100.0, 110.0, 120.0],
            "router_position": [0.1, 0.2, 0.3],
        }
    )

    metrics = daily_path_metrics(daily)

    assert round(metrics["avg_pos"], 2) == 0.2


def test_max_drawdown_interval_metrics_reports_peak_trough_position_and_trades():
    daily = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=6, freq="B"),
            "strategy_nav": [100.0, 120.0, 114.0, 90.0, 96.0, 130.0],
            "buyhold_nav": [100.0, 110.0, 105.0, 104.0, 106.0, 112.0],
            "position": [0.0, 0.7, 0.5, 0.3, 0.4, 0.2],
            "action": ["HOLD", "BUY", "HOLD", "SELL", "HOLD", "BUY"],
        }
    )

    metrics = max_drawdown_interval_metrics(daily)

    assert metrics["peak_date"] == "2026-01-02"
    assert metrics["trough_date"] == "2026-01-06"
    assert round(metrics["max_dd"], 2) == 25.0
    assert metrics["interval_days"] == 3
    assert round(metrics["interval_avg_pos"], 2) == 0.5
    assert metrics["interval_trades"] == 2
