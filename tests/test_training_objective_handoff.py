from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.soxl_quant.app.training_objective_handoff import derive_training_objective_handoff


def test_handoff_escalates_to_training_objective_after_path_router_failure():
    walkforward_scores = pd.DataFrame(
        [
            {
                "candidate": "recovery_reentry_conditional_vol_target_v1",
                "full_gate": False,
                "weighted_objective_loss": 237.49,
                "min_relative_return": 55.80,
                "max_max_dd": 26.14,
                "dominant_failure_type": "drawdown_control",
                "worst_window": "wf_2020",
            }
        ]
    )
    poor_window_clusters = pd.DataFrame(
        [
            {
                "cluster_id": "C2026",
                "cluster_start": "2026-02-17",
                "cluster_end": "2026-05-15",
                "dominant_failure_type": "underparticipation",
                "mean_excess_return": -167.30,
                "max_max_dd": 24.12,
                "mean_avg_pos": 0.09,
                "primary_optimization_path": "reentry_participation_or_behavior_clone",
            },
            {
                "cluster_id": "C2025",
                "cluster_start": "2025-04-16",
                "cluster_end": "2025-07-17",
                "dominant_failure_type": "underparticipation",
                "mean_excess_return": -160.50,
                "max_max_dd": 16.32,
                "mean_avg_pos": 0.52,
                "primary_optimization_path": "reentry_participation_or_behavior_clone",
            },
        ]
    )
    feature_groups = pd.DataFrame(
        [
            {"feature_group": "volatility_path", "top_mentions": 30, "mean_abs_delta_z": 1.26},
            {"feature_group": "trend_rebound", "top_mentions": 12, "mean_abs_delta_z": 0.96},
        ]
    )
    router_selection = {
        "decision": "diagnostic_no_promotion",
        "full_pass_count": 0,
        "cluster_pass": True,
        "selected_cluster_comparison": [
            {"cluster": "cluster_2026_high_vol_rebound", "excess_delta_vs_base": 51.36}
        ],
        "conclusion": [
            "No refreshed current-coverage rebound/downtrend router clears the full strict plus walk-forward plus cluster gate."
        ],
    }

    handoff = derive_training_objective_handoff(
        walkforward_scores=walkforward_scores,
        poor_window_clusters=poor_window_clusters,
        feature_groups=feature_groups,
        router_selection=router_selection,
    )

    assert handoff["decision"] == "train_state_conditioned_multi_objective"
    assert handoff["promotion"] == "blocked"
    assert handoff["stop_mechanisms"] == ["path_level_threshold_router_search"]
    assert handoff["selected_diagnostic_candidate"] == "recovery_reentry_conditional_vol_target_v1"
    assert handoff["risk_gap"]["max_dd_gap"] == 11.14
    assert handoff["primary_underparticipation_cluster"]["cluster_id"] == "C2026"
    assert handoff["state_feature_groups"] == ["volatility_path", "trend_rebound"]
    assert "walkforward_drawdown_gap_penalty" in handoff["objective_terms"]
    assert "high_vol_rebound_participation_floor" in handoff["objective_terms"]
    json.dumps(handoff)


def test_handoff_is_json_serializable_with_pandas_scalar_inputs():
    walkforward_scores = pd.DataFrame(
        [
            {
                "candidate": "numpy_candidate",
                "full_gate": np.bool_(False),
                "weighted_objective_loss": np.float64(1.0),
                "min_relative_return": np.float64(20.0),
                "max_max_dd": np.float64(18.0),
                "dominant_failure_type": "drawdown_control",
                "worst_window": "wf_2020",
            }
        ]
    )
    clusters = pd.DataFrame(
        [
            {
                "cluster_id": "C1",
                "cluster_start": "2026-01-01",
                "cluster_end": "2026-03-31",
                "dominant_failure_type": "underparticipation",
                "mean_excess_return": np.float64(-50.0),
                "max_max_dd": np.float64(8.0),
                "mean_avg_pos": np.float64(0.2),
                "window_count": np.int64(3),
            }
        ]
    )

    handoff = derive_training_objective_handoff(
        walkforward_scores=walkforward_scores,
        poor_window_clusters=clusters,
    )

    json.dumps(handoff)


def test_handoff_returns_promotion_review_when_full_gate_exists():
    walkforward_scores = pd.DataFrame(
        [
            {
                "candidate": "passing_candidate",
                "full_gate": True,
                "weighted_objective_loss": 0.0,
                "min_relative_return": 35.0,
                "max_max_dd": 12.0,
                "dominant_failure_type": "pass",
                "worst_window": "wf_2022",
            }
        ]
    )

    handoff = derive_training_objective_handoff(
        walkforward_scores=walkforward_scores,
        poor_window_clusters=pd.DataFrame(),
    )

    assert handoff["decision"] == "candidate_promotion_review"
    assert handoff["promotion"] == "review_required"
    assert handoff["selected_diagnostic_candidate"] == "passing_candidate"
