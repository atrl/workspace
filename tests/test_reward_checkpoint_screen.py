from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.reward_checkpoint_screen import (
    RewardCheckpointScreenConfig,
    render_reward_checkpoint_screen_review,
    score_reward_checkpoint_candidates,
    summarize_reward_checkpoint_screen,
)


def test_reward_checkpoint_screen_blocks_short_volshock_joint_tradeoff_before_ppo():
    gate_gaps = pd.DataFrame(
        [
            {
                "profile": "base_multi_obj_plain",
                "window": "wf_2022",
                "relative_return": 51.94,
                "max_dd": 36.33,
                "relative_gap": 0.0,
                "max_dd_gap": 21.33,
                "gate_pass": False,
                "failure_type": "drawdown_control",
            },
            {
                "profile": "short_volshock_plain",
                "window": "wf_2022",
                "relative_return": 67.62,
                "max_dd": 22.26,
                "relative_gap": 0.0,
                "max_dd_gap": 7.26,
                "gate_pass": False,
                "failure_type": "drawdown_control",
            },
            {
                "profile": "short_volshock_plain",
                "window": "wf_2023_2024",
                "relative_return": 5.02,
                "max_dd": 16.69,
                "relative_gap": 24.98,
                "max_dd_gap": 1.69,
                "gate_pass": False,
                "failure_type": "joint_tradeoff",
            },
            {
                "profile": "short_volshock_weighted",
                "window": "wf_2022",
                "relative_return": 65.87,
                "max_dd": 24.75,
                "relative_gap": 0.0,
                "max_dd_gap": 9.75,
                "gate_pass": False,
                "failure_type": "drawdown_control",
            },
            {
                "profile": "short_volshock_weighted",
                "window": "wf_2023_2024",
                "relative_return": 8.82,
                "max_dd": 12.80,
                "relative_gap": 21.18,
                "max_dd_gap": 0.0,
                "gate_pass": False,
                "failure_type": "selection_bias",
            },
        ]
    )
    slice_scores = pd.DataFrame(
        [
            {
                "candidate": "short_volshock_plain",
                "mean_short_volshock_accuracy": 0.96,
                "short_volshock_target_samples": 468,
                "slice_gate_pass_count": 2,
            },
            {
                "candidate": "short_volshock_weighted",
                "mean_short_volshock_accuracy": 0.97,
                "short_volshock_target_samples": 468,
                "slice_gate_pass_count": 2,
            },
        ]
    )
    clusters = pd.DataFrame(
        [
            {"cluster_id": "C004", "dominant_failure_type": "underparticipation"},
            {"cluster_id": "C002", "dominant_failure_type": "underparticipation"},
        ]
    )
    feature_groups = pd.DataFrame(
        [
            {"feature_group": "trend_quality", "max_stability_score": 2.43},
            {"feature_group": "volatility_regime", "max_stability_score": 2.32},
            {"feature_group": "drawdown_path", "max_stability_score": 1.84},
        ]
    )

    scored = score_reward_checkpoint_candidates(
        gate_gaps,
        slice_scores=slice_scores,
        poor_window_clusters=clusters,
        feature_groups=feature_groups,
        reference_candidate="base_multi_obj_plain",
        config=RewardCheckpointScreenConfig(max_dd_gap_weight=10.0),
    )
    summary = summarize_reward_checkpoint_screen(scored)
    review = render_reward_checkpoint_screen_review(scored, summary)

    assert scored.iloc[0]["candidate"] == "short_volshock_plain"
    assert bool(scored.iloc[0]["full_gate"]) is False
    assert summary["decision"] == "reward_checkpoint_screen_no_promotion"
    assert summary["selected_candidate"] == "short_volshock_plain"
    assert summary["ppo_allowed_next"] is False
    assert "drawdown_drift_penalty" in summary["required_objective_terms"]
    assert "trend_preservation_floor" in summary["required_objective_terms"]
    assert "short_volshock_state_deleverage_consistency" in summary["required_objective_terms"]
    assert "checkpoint_objective_loss" in summary["required_objective_terms"]
    assert "short_volshock_plain" in review


def test_reward_checkpoint_screen_promotes_only_when_all_windows_pass():
    gate_gaps = pd.DataFrame(
        [
            {
                "profile": "passing_candidate",
                "window": "wf_2022",
                "relative_return": 35.0,
                "max_dd": 12.0,
                "relative_gap": 0.0,
                "max_dd_gap": 0.0,
                "gate_pass": True,
                "failure_type": "pass",
            },
            {
                "profile": "passing_candidate",
                "window": "wf_2023_2024",
                "relative_return": 32.0,
                "max_dd": 13.0,
                "relative_gap": 0.0,
                "max_dd_gap": 0.0,
                "gate_pass": True,
                "failure_type": "pass",
            },
        ]
    )

    scored = score_reward_checkpoint_candidates(gate_gaps)
    summary = summarize_reward_checkpoint_screen(scored)

    assert bool(scored.iloc[0]["full_gate"]) is True
    assert scored.iloc[0]["reward_checkpoint_loss"] == 0.0
    assert summary["decision"] == "reward_checkpoint_promotion_review"
    assert summary["ppo_allowed_next"] is True
