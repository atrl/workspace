from __future__ import annotations

import numpy as np
import pandas as pd

from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v2 import (
    _action_grid,
    _direct_action_labels,
    _encode_action_labels,
    _normalize_weights,
    _simulate_path,
    _stitched_goal_summary,
)


def test_direct_action_label_encoding_is_contiguous_for_xgb_multiclass():
    labels = pd.Series([0, 2, 6, 6, 2, 0, 5])

    encoder, encoded = _encode_action_labels(labels)

    assert encoded.tolist() == [0, 1, 3, 3, 1, 0, 2]
    assert encoder.inverse_transform(np.arange(len(encoder.classes_))).tolist() == [0, 2, 5, 6]


def test_direct_action_labels_choose_best_open_to_open_action():
    frame = pd.DataFrame(
        {
            "etf_open_to_open_return": [0.09, -0.10],
            "underlying_open_to_open_return": [0.03, -0.03],
        }
    )

    labels = _direct_action_labels(frame, actions=_action_grid(), downside_penalty=0.0)

    assert labels.tolist() == [6, 0]


def test_simulate_path_normalizes_soxx_soxl_and_applies_drawdown_stop():
    frame = pd.DataFrame(
        {
            "etf_open_to_open_return": [0.10, -0.20, 0.10],
            "underlying_open_to_open_return": [0.03, -0.06, 0.03],
        }
    )
    soxx, soxl = _normalize_weights(np.array([0.8, 0.8, 0.8]), np.array([0.8, 0.8, 0.8]))

    path = _simulate_path(
        frame,
        soxx,
        soxl,
        scale=1.0,
        stop_thr=-0.05,
        stop_mult=0.0,
        cost_per_turnover=0.0,
    )

    assert np.all(path["soxx_weight"] + path["soxl_weight"] <= 1.0 + 1e-12)
    assert path["soxx_weight"][2] == 0.0
    assert path["soxl_weight"][2] == 0.0


def test_stitched_goal_requires_both_soxx_excess_and_ten_pct_drawdown():
    frontier = pd.DataFrame(
        [
            {"candidate": "high_return_high_dd", "return_pct": 500.0, "max_drawdown_pct": -25.0},
            {"candidate": "low_dd_low_return", "return_pct": 120.0, "max_drawdown_pct": -8.0},
        ]
    )

    goal = _stitched_goal_summary(frontier, benchmark={"return_pct": 413.0, "max_drawdown_pct": -43.0})

    assert goal["required_policy_satisfied"] is False
    assert goal["passes_target_count"] == 0
    assert goal["best_drawdown_safe_return_pct"] == 120.0
    assert goal["min_drawdown_if_excess_passes_pct"] == -25.0
    assert "no_candidate_combines_soxx_plus_20_with_10pct_drawdown" in goal["blocking_reasons"]
