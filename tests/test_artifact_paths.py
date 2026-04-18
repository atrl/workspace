from __future__ import annotations

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


if __name__ == "__main__":
    test_default_artifact_paths_stay_canonical()
    test_namespaced_artifact_paths_do_not_clobber_main_profile_outputs()
    test_artifact_tag_is_sanitized_for_worker_names()
    print("test_artifact_paths: OK")
