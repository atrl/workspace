from __future__ import annotations

import numpy as np
import pandas as pd

from src.soxl_quant.app.representation_diagnostics import (
    rank_auc_score,
    score_binary_representation_feature_sets,
)


def test_rank_auc_score_handles_perfect_and_reversed_ordering():
    labels = np.asarray([0, 0, 1, 1], dtype=bool)

    assert rank_auc_score(labels, np.asarray([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert rank_auc_score(labels, np.asarray([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert np.isnan(rank_auc_score(np.asarray([1, 1], dtype=bool), np.asarray([0.1, 0.2])))


def test_score_binary_representation_feature_sets_prefers_cross_fold_stable_features():
    frame = pd.DataFrame(
        {
            "fold": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "is_rebound": [False, False, True, True, False, False, True, True],
            "stable_feature": [-2.0, -1.0, 1.0, 2.0, -2.2, -1.2, 1.1, 2.1],
            "fold_leak_feature": [-2.0, -1.0, 1.0, 2.0, 2.2, 1.2, -1.1, -2.1],
            "noise_feature": [0.1, -0.1, 0.0, 0.2, -0.2, 0.2, -0.1, 0.1],
        }
    )

    result = score_binary_representation_feature_sets(
        frame,
        label_col="is_rebound",
        fold_col="fold",
        feature_sets={
            "stable": ["stable_feature"],
            "unstable": ["fold_leak_feature"],
            "mixed": ["stable_feature", "noise_feature"],
        },
        min_positive_eval=1,
        min_holdout_auc=0.90,
    )

    scores = result.feature_set_scores.set_index("feature_set")
    assert bool(scores.loc["stable", "representation_gate"]) is True
    assert scores.loc["stable", "holdout_auc_min"] == 1.0
    assert bool(scores.loc["unstable", "representation_gate"]) is False
    assert scores.loc["unstable", "holdout_auc_min"] == 0.0
    assert result.summary["best_feature_set"] == "stable"


def test_score_binary_representation_feature_sets_marks_missing_holdout_coverage():
    frame = pd.DataFrame(
        {
            "fold": ["a", "a", "b", "b"],
            "is_rebound": [False, True, False, False],
            "feature": [0.0, 1.0, 0.1, 0.2],
        }
    )

    result = score_binary_representation_feature_sets(
        frame,
        label_col="is_rebound",
        fold_col="fold",
        feature_sets={"feature": ["feature"]},
        min_positive_eval=1,
    )

    score = result.feature_set_scores.iloc[0]
    assert bool(score["coverage_gate"]) is False
    assert bool(score["representation_gate"]) is False
    assert result.fold_metrics.loc[result.fold_metrics["eval_fold"] == "b", "eval_positive"].iloc[0] == 0
