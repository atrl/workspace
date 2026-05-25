from __future__ import annotations

import numpy as np
import pandas as pd

from src.soxl_quant.app.representation_diagnostics import (
    StateHeadTargetRecipeConfig,
    rank_auc_score,
    score_binary_representation_feature_sets,
    score_state_head_target_recipes,
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


def test_score_state_head_target_recipes_routes_stable_state_to_target():
    frame = pd.DataFrame(
        {
            "fold": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "is_rebound": [False, False, True, True, False, False, True, True],
            "stable_feature": [-2.0, -1.0, 1.0, 2.0, -2.2, -1.2, 1.1, 2.1],
            "unstable_feature": [-2.0, -1.0, 1.0, 2.0, 2.2, 1.2, -1.1, -2.1],
            "true_target": [0.1, 0.1, 0.7, 0.7, 0.1, 0.1, 0.7, 0.7],
            "default_target": [0.1] * 8,
            "rebound_target": [0.7] * 8,
        }
    )

    result = score_state_head_target_recipes(
        frame,
        label_col="is_rebound",
        fold_col="fold",
        true_target_col="true_target",
        default_target_col="default_target",
        state_target_col="rebound_target",
        feature_sets={
            "stable": ["stable_feature"],
            "unstable": ["unstable_feature"],
        },
        thresholds=(0.0,),
        config=StateHeadTargetRecipeConfig(
            min_recall=0.95,
            min_precision=0.95,
            max_false_positive_rate=0.05,
            max_rebound_mae=0.01,
        ),
    )

    scores = result.recipe_scores.set_index("feature_set")
    assert bool(scores.loc["stable", "recipe_gate"]) is True
    assert scores.loc["stable", "state_recall"] == 1.0
    assert scores.loc["stable", "rebound_target_mae"] == 0.0
    assert bool(scores.loc["unstable", "recipe_gate"]) is False
    assert scores.loc["unstable", "state_recall"] == 0.0
    assert result.summary["best_feature_set"] == "stable"


def test_score_state_head_target_recipes_rejects_false_positive_lift():
    frame = pd.DataFrame(
        {
            "fold": ["a", "a", "b", "b"],
            "is_rebound": [False, True, False, True],
            "feature": [1.0, 2.0, 1.2, 2.2],
            "true_target": [0.1, 0.7, 0.1, 0.7],
            "default_target": [0.1] * 4,
            "rebound_target": [0.7] * 4,
        }
    )

    result = score_state_head_target_recipes(
        frame,
        label_col="is_rebound",
        fold_col="fold",
        true_target_col="true_target",
        default_target_col="default_target",
        state_target_col="rebound_target",
        feature_sets={"loose": ["feature"]},
        thresholds=(-10.0,),
        config=StateHeadTargetRecipeConfig(max_false_positive_rate=0.0),
    )

    row = result.recipe_scores.iloc[0]
    assert row["false_positive_rate"] == 1.0
    assert row["effective_false_positive_rate"] == 1.0
    assert row["false_positive_lift_mean"] == 0.6
    assert bool(row["recipe_gate"]) is False


def test_state_head_recipe_ignores_false_positive_when_target_is_unchanged():
    frame = pd.DataFrame(
        {
            "fold": ["a", "a", "b", "b"],
            "is_rebound": [False, True, False, True],
            "feature": [1.0, 2.0, 1.2, 2.2],
            "true_target": [0.4, 0.7, 0.4, 0.7],
            "default_target": [0.4, 0.1, 0.4, 0.1],
            "rebound_target": [0.4, 0.7, 0.4, 0.7],
        }
    )

    result = score_state_head_target_recipes(
        frame,
        label_col="is_rebound",
        fold_col="fold",
        true_target_col="true_target",
        default_target_col="default_target",
        state_target_col="rebound_target",
        feature_sets={"loose": ["feature"]},
        thresholds=(-10.0,),
        config=StateHeadTargetRecipeConfig(
            min_recall=1.0,
            min_precision=0.0,
            max_false_positive_rate=1.0,
            max_effective_false_positive_rate=0.0,
            max_false_positive_lift_mean=0.0,
            max_false_positive_lift_max=0.0,
            max_target_mae=0.0,
            max_rebound_mae=0.0,
        ),
    )

    row = result.recipe_scores.iloc[0]
    assert row["false_positive_rate"] == 1.0
    assert row["effective_false_positive_rate"] == 0.0
    assert row["false_positive_lift_max"] == 0.0
    assert bool(row["recipe_gate"]) is True
