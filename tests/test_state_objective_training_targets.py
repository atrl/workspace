from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.state_objective_training_targets import (
    StateObjectiveTrainingTargetConfig,
    build_state_objective_training_targets,
    summarize_state_objective_training_targets,
)


def test_training_targets_mark_rebound_cluster_and_floor_teacher_position():
    daily_paths = {
        "strict_oos": pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-02-01", "2026-02-03", "2026-02-04"]),
                "strategy_nav": [100.0, 101.0, 102.0],
                "buyhold_nav": [100.0, 120.0, 130.0],
                "router_position": [0.12, 0.18, 0.42],
                "rebound_state": [False, True, True],
                "risk_state": [False, False, False],
            }
        )
    }
    window_metrics = pd.DataFrame(
        {
            "candidate": ["teacher"],
            "window": ["strict_oos"],
            "relative_return": [-45.0],
            "max_dd": [12.0],
        }
    )
    cluster_metrics = pd.DataFrame(
        {
            "candidate": ["teacher"],
            "cluster": ["cluster_2026_high_vol_rebound"],
            "start": ["2026-02-03"],
            "end": ["2026-02-04"],
            "relative_return": [-140.0],
            "max_dd": [8.0],
            "avg_pos": [0.20],
        }
    )

    targets = build_state_objective_training_targets(
        daily_paths,
        window_metrics=window_metrics,
        cluster_metrics=cluster_metrics,
        candidate="teacher",
    )

    cluster_rows = targets[targets["cluster"] == "cluster_2026_high_vol_rebound"]
    outside_rows = targets[targets["cluster"] == ""]
    assert cluster_rows["is_rebound_cluster"].all()
    assert cluster_rows["rebound_participation_weight"].min() > 0
    assert cluster_rows["rebound_target_position"].iloc[0] == 0.35
    assert cluster_rows["rebound_target_position"].iloc[1] == 0.42
    assert outside_rows["rebound_participation_weight"].max() == 0.0
    assert outside_rows["rebound_target_position"].iloc[0] == 0.12


def test_training_targets_mark_downtrend_drawdown_cap_from_window_gap():
    daily_paths = {
        "wf_2020": pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2020-03-02", "2020-03-03"]),
                "strategy_nav": [100.0, 90.0],
                "buyhold_nav": [100.0, 80.0],
                "position": [0.62, 0.55],
                "rebound_state": [False, False],
                "risk_state": [True, False],
            }
        )
    }
    window_metrics = pd.DataFrame(
        {
            "candidate": ["teacher"],
            "window": ["wf_2020"],
            "relative_return": [40.0],
            "max_dd": [27.0],
        }
    )
    cluster_metrics = pd.DataFrame(
        columns=["candidate", "cluster", "start", "end", "relative_return", "max_dd", "avg_pos"]
    )

    targets = build_state_objective_training_targets(
        daily_paths,
        window_metrics=window_metrics,
        cluster_metrics=cluster_metrics,
        candidate="teacher",
        config=StateObjectiveTrainingTargetConfig(downtrend_position_cap=0.25),
    )

    risk_row = targets.loc[targets["is_downtrend_state"]].iloc[0]
    normal_row = targets.loc[~targets["is_downtrend_state"]].iloc[0]
    assert risk_row["downtrend_drawdown_weight"] > 0
    assert risk_row["risk_target_position"] == 0.25
    assert normal_row["downtrend_drawdown_weight"] == 0.0
    assert normal_row["risk_target_position"] == 0.55


def test_training_target_summary_reports_top_weighted_windows_and_terms():
    daily_paths = {
        "wf_2020": pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                "strategy_nav": [100.0, 101.0],
                "buyhold_nav": [100.0, 120.0],
                "position": [0.10, 0.15],
                "rebound_state": [False, True],
                "risk_state": [False, False],
            }
        )
    }
    window_metrics = pd.DataFrame(
        {
            "candidate": ["teacher"],
            "window": ["wf_2020"],
            "relative_return": [-20.0],
            "max_dd": [18.0],
        }
    )
    cluster_metrics = pd.DataFrame(
        {
            "candidate": ["teacher"],
            "cluster": ["cluster_a"],
            "start": ["2020-01-03"],
            "end": ["2020-01-03"],
            "relative_return": [-70.0],
            "max_dd": [5.0],
            "avg_pos": [0.15],
        }
    )
    targets = build_state_objective_training_targets(
        daily_paths,
        window_metrics=window_metrics,
        cluster_metrics=cluster_metrics,
        candidate="teacher",
    )

    summary = summarize_state_objective_training_targets(targets)

    assert summary["rows"] == 2
    assert summary["top_weighted_window"] == "wf_2020"
    assert summary["rebound_rows"] == 1
    assert summary["downtrend_rows"] == 0
    assert "high_vol_rebound_participation_floor" in summary["required_terms"]
    assert "walkforward_drawdown_gap_penalty" in summary["required_terms"]
