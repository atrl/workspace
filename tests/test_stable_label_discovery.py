from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.stable_label_discovery import (
    BinaryLabelDiscoveryConfig,
    score_binary_label_candidates,
)


def test_score_binary_label_candidates_prefers_stable_forward_lift_label():
    frame = pd.DataFrame(
        {
            "datetime": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-02",
                "2020-01-03",
                "2020-01-03",
            ],
            "fold": ["a", "b", "a", "b", "a", "b"],
            "forward_return_5": [-0.03, -0.02, 0.12, 0.10, 0.09, 0.11],
            "target_lift": [0.0, 0.0, 0.20, 0.22, 0.18, 0.19],
        }
    )
    candidates = {
        "stable_rebound": pd.Series([False, False, True, True, True, True]),
        "inactive": pd.Series([False, False, False, False, False, False]),
    }

    result = score_binary_label_candidates(
        frame,
        candidates,
        group_col="datetime",
        fold_col="fold",
        forward_return_cols=("forward_return_5",),
        target_lift_col="target_lift",
        config=BinaryLabelDiscoveryConfig(
            min_positive_rows=2,
            min_fold_coverage=2,
            min_fold_positive_rows=1,
            min_forward_return_lift=0.05,
            min_target_lift=0.05,
            max_positive_rate=0.90,
        ),
    )

    scores = result.candidate_scores.set_index("candidate")
    assert bool(scores.loc["stable_rebound", "label_gate"]) is True
    assert scores.loc["stable_rebound", "positive_rows"] == 4
    assert scores.loc["stable_rebound", "label_conflict_groups"] == 0
    assert scores.loc["stable_rebound", "primary_forward_return_lift"] > 0.10
    assert scores.loc["stable_rebound", "target_lift_mean"] > 0.15
    assert bool(scores.loc["inactive", "label_gate"]) is False
    assert result.summary["selected_candidate"] == "stable_rebound"


def test_score_binary_label_candidates_rejects_duplicate_date_conflicts():
    frame = pd.DataFrame(
        {
            "datetime": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "fold": ["a", "b", "a", "b"],
            "forward_return_5": [0.10, 0.10, 0.11, 0.11],
            "target_lift": [0.20, 0.20, 0.21, 0.21],
        }
    )
    candidates = {
        "conflicting_rebound": pd.Series([True, False, True, True]),
    }

    result = score_binary_label_candidates(
        frame,
        candidates,
        group_col="datetime",
        fold_col="fold",
        forward_return_cols=("forward_return_5",),
        target_lift_col="target_lift",
        config=BinaryLabelDiscoveryConfig(
            min_positive_rows=2,
            min_fold_coverage=2,
            min_fold_positive_rows=1,
            min_forward_return_lift=0.0,
            min_target_lift=0.0,
            max_label_conflict_rate=0.0,
        ),
    )

    score = result.candidate_scores.iloc[0]
    assert score["label_conflict_groups"] == 1
    assert score["label_conflict_rate"] == 0.5
    assert bool(score["label_gate"]) is False
    assert result.summary["label_gate"] is False
