from pathlib import Path

import pandas as pd

from src.workflow.stage1_feature_recovery_ablation import (
    ABSOLUTE_RISK_FEATURES,
    BREADTH_FEATURES,
    FEATURE_SETS,
    Stage1FeatureRecoveryAblationSpec,
    annotate_candidate_neighborhood,
    select_feature_recovery_candidate,
)


def test_feature_sets_preserve_baseline_and_add_causal_breadth() -> None:
    baseline = FEATURE_SETS["baseline_stage1"]
    enhanced = FEATURE_SETS["absolute_plus_breadth"]
    assert set(baseline).issubset(enhanced)
    assert set(ABSOLUTE_RISK_FEATURES).issubset(enhanced)
    assert set(BREADTH_FEATURES).issubset(enhanced)
    assert all("lag1" in feature for feature in BREADTH_FEATURES)


def test_candidate_neighborhood_is_feature_set_local() -> None:
    candidates = pd.DataFrame(
        [
            {
                "feature_set": feature_set,
                "logistic_c": 0.1,
                "probability_threshold": threshold,
                "action_beta": 0.6,
                "terminal_wealth": wealth,
                "development_gate_pass": True,
            }
            for feature_set, wealth in (
                ("baseline_stage1", 5.0),
                ("absolute_plus_breadth", 5.3),
            )
            for threshold in (0.65, 0.70, 0.75)
        ]
    )
    result = annotate_candidate_neighborhood(candidates)
    selected = result[
        result["feature_set"].eq("absolute_plus_breadth")
        & result["probability_threshold"].eq(0.70)
    ].iloc[0]
    assert selected["neighbor_count"] == 3
    assert selected["neighbor_min_wealth_ratio"] == 1.0


def test_selection_uses_development_only_and_requires_real_gain() -> None:
    spec = Stage1FeatureRecoveryAblationSpec(root=Path("."))
    common = {
        "segment": "development_walk_forward",
        "selection_uses_outer_test": False,
        "logistic_c": 0.03,
        "probability_threshold": 0.55,
        "action_beta": 0.6,
        "development_gate_pass": True,
        "relative_ratio_p25": 1.05,
        "neighbor_count": 5,
        "neighbor_min_wealth_ratio": 0.95,
    }
    candidates = pd.DataFrame(
        [
            {
                **common,
                "feature_set": "baseline_stage1",
                "terminal_wealth": 5.0,
                "total_turnover": 100.0,
            },
            {
                **common,
                "feature_set": "absolute_plus_breadth",
                "terminal_wealth": 5.2,
                "total_turnover": 90.0,
            },
            {
                **common,
                "segment": "outer_diagnostic",
                "selection_uses_outer_test": True,
                "feature_set": "absolute_plus_breadth",
                "terminal_wealth": 50.0,
                "total_turnover": 1.0,
            },
        ]
    )
    selected = select_feature_recovery_candidate(
        candidates,
        spec=spec,
    )
    assert selected["feature_set"] == "absolute_plus_breadth"
    assert selected["segment"] == "development_walk_forward"
