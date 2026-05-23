from __future__ import annotations

from scripts import prune_artifacts


def test_runtime_actor_path_resolves_from_profile_directory():
    path = prune_artifacts.ROOT / (
        "runtime/model/card/balanced_rule_residual_v22/"
        "soxl_balanced_rule_residual_v22_actor.pt"
    )

    assert prune_artifacts._profile_name_from_path(path, allow_summary_namespace=False) == "balanced_rule_residual_v22"


def test_walkforward_actor_path_resolves_to_base_profile():
    path = prune_artifacts.ROOT / (
        "runtime/model/card/balanced_rule_residual_v22_wf_2020/"
        "soxl_balanced_rule_residual_v22_wf_2020_actor.pt"
    )

    assert prune_artifacts._profile_name_from_path(path, allow_summary_namespace=False) == "balanced_rule_residual_v22"


def test_checkpoint_actor_path_uses_retention_profile_directory():
    path = prune_artifacts.ROOT / (
        "runtime/model/card/balanced_rule_residual_v22_continued_retention_v1/"
        "soxl_balanced_rule_residual_v22_continued_retention_v1_checkpoint_iter10_actor.pt"
    )

    assert (
        prune_artifacts._profile_name_from_path(path, allow_summary_namespace=False)
        == "balanced_rule_residual_v22_continued_retention_v1"
    )
