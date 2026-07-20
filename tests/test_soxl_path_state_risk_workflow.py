from __future__ import annotations

import pandas as pd


def test_segment_purge_blocks_forward_paths_from_crossing_boundaries():
    from src.workflow.soxl_path_state_risk import split_and_purge_segments

    dates = pd.bdate_range("2022-11-01", "2025-02-28")
    frame = pd.DataFrame(
        {
            "datetime": dates,
            "fast_tail_5d": 0.0,
            "structural_break_20d": 0.0,
            "forward_terminal_5": 0.0,
            "forward_mae_5": 0.0,
            "forward_mfe_5": 0.0,
            "forward_terminal_20": 0.0,
            "forward_mae_20": 0.0,
            "forward_mfe_20": 0.0,
        }
    )
    segments = split_and_purge_segments(
        frame,
        development=("2022-11-01", "2022-12-30"),
        validation=("2023-01-03", "2024-12-31"),
        test=("2025-01-02", "2025-02-28"),
    )
    for segment in segments.values():
        assert segment["fast_tail_5d"].tail(6).isna().all()
        assert segment["structural_break_20d"].tail(21).isna().all()


def test_episode_ids_split_at_segment_boundaries_but_preserve_causal_age():
    from src.workflow.soxl_path_state_risk import add_segment_episode_ids

    frame = pd.DataFrame(
        {
            "segment": ["development", "development", "validation", "validation"],
            "event": [True, True, True, True],
            "structural_chop_age": [3, 4, 5, 6],
        }
    )
    result = add_segment_episode_ids(frame, event_column="event")
    assert result["episode_id"].nunique() == 2
    assert result["structural_chop_age"].tolist() == [3, 4, 5, 6]


def test_retention_gate_rejects_direction_instability_and_rare_structural_support():
    from src.workflow.soxl_path_state_risk import evaluate_path_state_decision

    stable = {
        "development": {"observations": 60, "episodes": 21, "auc": 0.60, "quintile_lift": 0.08},
        "validation": {"observations": 20, "episodes": 4, "auc": 0.58, "quintile_lift": 0.06, "conditional_lift": 0.02},
        "test": {"observations": 18, "episodes": 4, "auc": 0.56, "quintile_lift": 0.04, "conditional_lift": 0.01},
    }
    reversed_test = {**stable, "test": {**stable["test"], "auc": 0.44, "quintile_lift": -0.02}}
    assert evaluate_path_state_decision("fast", stable)["decision"] == "retained_incremental_factor_evidence"
    assert evaluate_path_state_decision("fast", reversed_test)["decision"] == "not_retained_direction_instability"

    rare = {
        "development": {"observations": 80, "episodes": 20, "auc": 0.61, "quintile_lift": 0.10},
        "validation": {"observations": 12, "episodes": 3, "auc": 0.62, "quintile_lift": 0.12, "conditional_lift": 0.03},
        "test": {"observations": 8, "episodes": 2, "auc": 0.70, "quintile_lift": 0.20, "conditional_lift": 0.05},
    }
    assert evaluate_path_state_decision("structural", rare)["decision"] == "diagnostic_rare_event_support"
