from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from src.soxl_quant.config import load_profile
from src.soxl_quant.app.short_foldlocal_training_screen import (
    apply_short_training_patch,
    build_short_foldlocal_training_plan,
    resolve_local_checkpoint_path,
    summarize_short_foldlocal_training_plan,
)


def test_resolve_local_checkpoint_path_rewrites_stale_absolute_summary_path(tmp_path: Path):
    card = tmp_path / "runtime" / "model" / "card" / "candidate"
    card.mkdir(parents=True)
    local = card / "soxl_candidate_wf_2020_checkpoint_iter8_actor.pt"
    local.write_bytes(b"checkpoint")
    stale = (
        "/Users/rziiiii/.openclaw/workspace/runtime/model/card/candidate/"
        "soxl_candidate_wf_2020_checkpoint_iter8_actor.pt"
    )

    resolved = resolve_local_checkpoint_path(stale, source_model_card=card)

    assert resolved == local


def test_apply_short_training_patch_keeps_source_profile_unchanged_and_disables_promotion():
    source = load_profile("mode_conditioned_rebound_reward_train_v2")
    original_reward = asdict(source.reward)

    patched = apply_short_training_patch(
        source,
        training_profile_name="mode_conditioned_drawdown_drift_short_train_v1",
        reward_patch={
            "drawdown_drift_reward_enabled": True,
            "drawdown_drift_penalty": 1.5,
            "trend_preservation_floor_enabled": True,
        },
        checkpoint_path="runtime/model/card/source_actor.pt",
        max_iters=4,
        stage_iters=2,
    )

    assert source.name == "mode_conditioned_rebound_reward_train_v2"
    assert asdict(source.reward) == original_reward
    assert patched.name == "mode_conditioned_drawdown_drift_short_train_v1"
    assert patched.max_iters == 4
    assert patched.selection_score.stage_iters == 2
    assert patched.checkpoint_path == "runtime/model/card/source_actor.pt"
    assert patched.reward.drawdown_drift_reward_enabled is True
    assert patched.reward.drawdown_drift_penalty == 1.5
    assert patched.reward.trend_preservation_floor_enabled is True
    assert patched.promotion_gate.max_test_2022_drawdown == source.promotion_gate.max_test_2022_drawdown


def test_apply_short_training_patch_rejects_unknown_reward_field():
    source = load_profile("mode_conditioned_rebound_reward_train_v2")

    with pytest.raises(ValueError, match="unknown reward patch field"):
        apply_short_training_patch(
            source,
            training_profile_name="bad_short_train",
            reward_patch={"not_a_reward_field": 1.0},
            checkpoint_path="runtime/model/card/source_actor.pt",
        )


def test_build_short_foldlocal_training_plan_uses_current_workspace_paths(tmp_path: Path):
    source = load_profile("mode_conditioned_rebound_reward_train_v2")
    card = tmp_path / "runtime" / "model" / "card" / "candidate"
    card.mkdir(parents=True)
    for fold in ("wf_2020", "wf_2022"):
        (card / f"soxl_{source.name}_{fold}_checkpoint_iter8_actor.pt").write_bytes(b"checkpoint")

    plan = build_short_foldlocal_training_plan(
        source_profile=source,
        source_model_card=card,
        training_profile_name="mode_conditioned_drawdown_drift_short_train_v1",
        reward_patch={"drawdown_drift_reward_enabled": True},
        required_folds=("wf_2020", "wf_2022"),
        max_iters=4,
        stage_iters=2,
    )
    summary = summarize_short_foldlocal_training_plan(plan)

    assert summary["decision"] == "short_foldlocal_training_plan_ready_no_promotion"
    assert summary["training_allowed"] is True
    assert summary["promotion_allowed"] is False
    assert set(plan["fold"]) == {"wf_2020", "wf_2022"}
    assert plan["source_checkpoint_exists"].all()
    assert plan["promotion_allowed"].eq(False).all()
    assert plan["training_profile"].eq("mode_conditioned_drawdown_drift_short_train_v1").all()
