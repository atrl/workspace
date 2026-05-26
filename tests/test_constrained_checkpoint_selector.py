from __future__ import annotations

import pandas as pd

from src.soxl_quant.app.constrained_checkpoint_selector import (
    ConstrainedCheckpointSelectorConfig,
    render_constrained_checkpoint_selection_review,
    score_constrained_checkpoints,
    summarize_constrained_checkpoint_selection,
)


def test_constrained_checkpoint_selector_prioritizes_full_gate_over_objective_score():
    gate_metrics = pd.DataFrame(
        [
            {
                "candidate": "validation_winner_breach",
                "window": "wf_2022",
                "relative_return": 72.0,
                "max_dd": 23.0,
            },
            {
                "candidate": "validation_winner_breach",
                "window": "wf_2023_2024",
                "relative_return": 44.0,
                "max_dd": 13.5,
            },
            {
                "candidate": "constraint_candidate",
                "window": "wf_2022",
                "relative_return": 34.0,
                "max_dd": 14.0,
            },
            {
                "candidate": "constraint_candidate",
                "window": "wf_2023_2024",
                "relative_return": 31.0,
                "max_dd": 14.5,
            },
        ]
    )
    checkpoint_scores = pd.DataFrame(
        [
            {
                "candidate": "validation_winner_breach",
                "alignment_loss": 0.01,
                "checkpoint_objective_loss": 0.10,
                "coverage_gate": True,
            },
            {
                "candidate": "constraint_candidate",
                "alignment_loss": 2.50,
                "checkpoint_objective_loss": 5.00,
                "coverage_gate": True,
            },
        ]
    )

    scored = score_constrained_checkpoints(gate_metrics, checkpoint_scores=checkpoint_scores)
    summary = summarize_constrained_checkpoint_selection(scored)
    review = render_constrained_checkpoint_selection_review(scored, summary)

    assert scored.iloc[0]["candidate"] == "constraint_candidate"
    assert bool(scored.iloc[0]["selector_gate_pass"]) is True
    assert bool(scored.iloc[1]["selector_gate_pass"]) is False
    assert summary["decision"] == "constrained_checkpoint_candidate_ready_for_replay_review"
    assert summary["selected_candidate"] == "constraint_candidate"
    assert summary["promotion_allowed"] is False
    assert summary["replay_review_required"] is True
    assert "constraint_candidate" in review


def test_constrained_checkpoint_selector_blocks_current_short_volshock_candidates():
    gate_metrics = pd.DataFrame(
        [
            {
                "profile": "base_multi_obj_plain",
                "window": "wf_2022",
                "relative_return": 51.935249857961985,
                "max_dd": 36.332029087228655,
            },
            {
                "profile": "base_multi_obj_plain",
                "window": "wf_2023_2024",
                "relative_return": 33.75925662569212,
                "max_dd": 20.907231007311154,
            },
            {
                "profile": "short_volshock_plain",
                "window": "wf_2022",
                "relative_return": 67.62439551395902,
                "max_dd": 22.26198899445525,
            },
            {
                "profile": "short_volshock_plain",
                "window": "wf_2023_2024",
                "relative_return": 5.018898654205939,
                "max_dd": 16.69308745121708,
            },
            {
                "profile": "short_volshock_weighted_obj025_cb099",
                "window": "wf_2022",
                "relative_return": 65.87292599288905,
                "max_dd": 24.745906422864827,
            },
            {
                "profile": "short_volshock_weighted_obj025_cb099",
                "window": "wf_2023_2024",
                "relative_return": 8.823051141847593,
                "max_dd": 12.799186790968776,
            },
        ]
    )
    checkpoint_scores = pd.DataFrame(
        [
            {
                "candidate": "short_volshock_plain",
                "alignment_loss": 0.1514566918828364,
                "checkpoint_objective_loss": 2.2650405491687104,
                "coverage_gate": True,
            },
            {
                "candidate": "short_volshock_weighted_obj025_cb099",
                "alignment_loss": 0.15545114258042558,
                "checkpoint_objective_loss": 2.6412654631637427,
                "coverage_gate": True,
            },
            {
                "candidate": "base_multi_obj_plain",
                "alignment_loss": 2.237350469858388,
                "checkpoint_objective_loss": 18.20554702427619,
                "coverage_gate": False,
                "missing_target_ratio": 1.0,
            },
        ]
    )

    scored = score_constrained_checkpoints(
        gate_metrics,
        checkpoint_scores=checkpoint_scores,
        reference_candidate="base_multi_obj_plain",
        config=ConstrainedCheckpointSelectorConfig(max_dd_gap_weight=10.0),
    )
    summary = summarize_constrained_checkpoint_selection(scored)

    assert bool(scored["selector_gate_pass"].any()) is False
    assert scored.iloc[0]["candidate"] == "short_volshock_plain"
    assert summary["decision"] == "constrained_checkpoint_selector_no_promotion"
    assert summary["selected_candidate"] == "short_volshock_plain"
    assert summary["promotion_allowed"] is False
    assert summary["replay_review_required"] is False
    assert summary["next_mechanism"] == "generate_reward_constrained_short_training_candidate"
