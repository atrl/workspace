from __future__ import annotations

from scripts.walkforward_backtest_validate import model_path_for_profile
from src.soxl_quant.settings import artifact_stem, model_artifact_path, summary_artifact_path


def test_default_artifact_paths_stay_canonical():
    assert artifact_stem("balanced") == "soxl_balanced"
    assert model_artifact_path("balanced").name == "soxl_balanced_actor.pt"
    assert summary_artifact_path("balanced").name == "soxl_balanced_summary.json"


def test_namespaced_artifact_paths_do_not_clobber_main_profile_outputs():
    tag = "wf_2023_2024"
    assert artifact_stem("balanced", tag) == "soxl_balanced_wf_2023_2024"
    assert model_artifact_path("balanced", tag).name == "soxl_balanced_wf_2023_2024_actor.pt"
    assert summary_artifact_path("balanced", tag).name == "soxl_balanced_wf_2023_2024_summary.json"
    assert model_artifact_path("balanced", tag) != model_artifact_path("balanced")
    assert summary_artifact_path("balanced", tag) != summary_artifact_path("balanced")


def test_artifact_tag_is_sanitized_for_worker_names():
    tag = "wf 2023/2024 final"
    assert artifact_stem("smoke", tag) == "soxl_smoke_wf_2023_2024_final"


def test_walkforward_validator_uses_canonical_profile_model_path():
    assert model_path_for_profile("balanced_rule_residual_v22") == model_artifact_path(
        "balanced_rule_residual_v22"
    )


if __name__ == "__main__":
    test_default_artifact_paths_stay_canonical()
    test_namespaced_artifact_paths_do_not_clobber_main_profile_outputs()
    test_artifact_tag_is_sanitized_for_worker_names()
    test_walkforward_validator_uses_canonical_profile_model_path()
    print("test_artifact_paths: OK")
