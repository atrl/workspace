from __future__ import annotations

import numpy as np
import pandas as pd

from src.soxl_quant.app.representation_diagnostics import (
    StateHeadConsensusTargetConfig,
    StateHeadLiftStabilityConfig,
    StateHeadTargetRecipeConfig,
    derive_state_head_consensus_targets,
    rank_auc_score,
    score_binary_representation_feature_sets,
    score_state_head_lift_stability,
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


def test_state_head_lift_stability_flags_fold_conflicting_effective_lifts():
    row_scores = pd.DataFrame(
        {
            "datetime": ["2020-06-03", "2020-06-03", "2021-04-05"],
            "is_rebound": [True, False, True],
            "predicted_state": [True, True, True],
            "target_lift": [0.20, 0.18, 0.15],
        }
    )

    result = score_state_head_lift_stability(
        row_scores,
        group_col="datetime",
        label_col="is_rebound",
        config=StateHeadLiftStabilityConfig(min_lift=0.01, max_unstable_effective_lift_rate=0.0),
    )

    summary = result.summary
    assert summary["effective_lift_rows"] == 3
    assert summary["unstable_effective_lift_rows"] == 2
    assert summary["false_positive_effective_lift_rows"] == 1
    assert summary["unstable_false_positive_effective_lift_rows"] == 1
    assert bool(summary["stability_gate"]) is False

    unstable = result.group_metrics.set_index("datetime").loc["2020-06-03"]
    assert bool(unstable["label_unstable"]) is True
    assert unstable["unstable_effective_lift_rows"] == 2


def test_state_head_lift_stability_passes_when_effective_lifts_are_label_stable():
    row_scores = pd.DataFrame(
        {
            "datetime": ["2020-06-03", "2020-06-03", "2021-04-05"],
            "is_rebound": [True, True, False],
            "predicted_state": [True, True, False],
            "target_lift": [0.20, 0.18, 0.0],
        }
    )

    result = score_state_head_lift_stability(
        row_scores,
        group_col="datetime",
        label_col="is_rebound",
        config=StateHeadLiftStabilityConfig(min_lift=0.01, max_unstable_effective_lift_rate=0.0),
    )

    assert result.summary["unstable_effective_lift_rows"] == 0
    assert result.summary["stability_gate"] is True


def test_state_head_lift_stability_passes_when_no_effective_lifts_exist():
    row_scores = pd.DataFrame(
        {
            "datetime": ["2020-06-03", "2020-06-03"],
            "is_rebound": [True, False],
            "predicted_state": [False, False],
            "target_lift": [0.0, 0.0],
        }
    )

    result = score_state_head_lift_stability(
        row_scores,
        group_col="datetime",
        label_col="is_rebound",
        config=StateHeadLiftStabilityConfig(min_lift=0.01, max_unstable_effective_lift_rate=0.0),
    )

    assert result.summary["effective_lift_rows"] == 0
    assert result.summary["unstable_effective_lift_rate"] == 0.0
    assert result.summary["stability_gate"] is True


def test_derive_state_head_consensus_targets_requires_fold_agreement():
    row_scores = pd.DataFrame(
        {
            "datetime": ["2020-06-03", "2020-06-03", "2021-04-05", "2021-04-05"],
            "is_rebound": [True, False, True, True],
            "predicted_state": [True, True, True, True],
            "default_target": [0.25, 0.25, 0.20, 0.21],
            "proposed_target": [0.45, 0.43, 0.35, 0.36],
            "target_lift": [0.20, 0.18, 0.15, 0.15],
        }
    )

    result = derive_state_head_consensus_targets(
        row_scores,
        group_col="datetime",
        label_col="is_rebound",
        default_target_col="default_target",
        proposed_target_col="proposed_target",
        config=StateHeadConsensusTargetConfig(min_sources=2, min_lift=0.01),
    )

    targets = result.consensus_targets.set_index("datetime")
    assert "2020-06-03" not in targets.index
    assert "2021-04-05" in targets.index
    assert targets.loc["2021-04-05", "consensus_lift"] == 0.15
    assert targets.loc["2021-04-05", "consensus_target_position"] == 0.355
    assert result.summary["consensus_target_groups"] == 1
    assert result.summary["conflict_groups"] == 1
    assert result.summary["consensus_gate"] is True


def test_derive_state_head_consensus_targets_blocks_when_no_stable_lift_survives():
    row_scores = pd.DataFrame(
        {
            "datetime": ["2020-06-03", "2020-06-03"],
            "is_rebound": [True, False],
            "predicted_state": [True, True],
            "default_target": [0.25, 0.25],
            "proposed_target": [0.45, 0.43],
            "target_lift": [0.20, 0.18],
        }
    )

    result = derive_state_head_consensus_targets(
        row_scores,
        group_col="datetime",
        label_col="is_rebound",
        default_target_col="default_target",
        proposed_target_col="proposed_target",
        config=StateHeadConsensusTargetConfig(min_sources=2, min_lift=0.01),
    )

    assert result.consensus_targets.empty
    assert result.summary["consensus_target_groups"] == 0
    assert result.summary["consensus_gate"] is False
