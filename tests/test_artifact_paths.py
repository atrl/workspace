from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import scripts.export_profile_daily_backtest as export_backtest
from scripts.export_profile_daily_backtest import load_profile_backtest_inputs, resolve_model_path
from scripts.walkforward_backtest_validate import model_path_for_profile
from src.soxl_quant.app import artifacts
from src.soxl_quant.config import load_profile
import src.soxl_quant.settings as settings
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


def test_export_profile_daily_backtest_resolves_saved_walkforward_actor_layout():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "card"
        actor = root / "balanced_rule_residual_v22_wf_2022" / "soxl_balanced_rule_residual_v22_wf_2022_actor.pt"
        actor.parent.mkdir(parents=True)
        actor.write_bytes(b"actor")

        assert (
            resolve_model_path(
                "balanced_rule_residual_v22",
                "wf_2022",
                model_dir=root,
            )
            == actor
        )


def test_export_profile_daily_backtest_prefers_explicit_model_path():
    with TemporaryDirectory() as tmpdir:
        explicit = Path(tmpdir) / "custom_actor.pt"
        explicit.write_bytes(b"actor")

        assert resolve_model_path("balanced", "wf_2020", explicit_model=explicit) == explicit


def test_export_profile_daily_backtest_uses_profile_specific_feature_packs():
    calls: dict[str, object] = {}

    class DummyHandler:
        feature_frame = None
        price_frame = None

    class DummyDataset:
        handler = DummyHandler()

    class DummyProfile:
        name = "custom_profile"

        class feature:
            packs = ["vix_term", "transition_signals"]

    def fake_build_dataset(*, feature_packs=(), profile_name=""):
        calls["feature_packs"] = feature_packs
        calls["profile_name"] = profile_name
        return DummyDataset()

    def fake_feature_columns_for_packs(packs, profile_name=""):
        calls["feature_columns_packs"] = tuple(packs)
        calls["feature_columns_profile_name"] = profile_name
        return ["rsi_6", "vix_level"]

    original_build_dataset = export_backtest.build_dataset
    original_feature_columns_for_packs = export_backtest.feature_columns_for_packs
    original_segment_frame = export_backtest.segment_frame
    try:
        export_backtest.build_dataset = fake_build_dataset
        export_backtest.feature_columns_for_packs = fake_feature_columns_for_packs
        export_backtest.segment_frame = lambda _dataset, _segment: "test_df"
        result = load_profile_backtest_inputs(DummyProfile())
    finally:
        export_backtest.build_dataset = original_build_dataset
        export_backtest.feature_columns_for_packs = original_feature_columns_for_packs
        export_backtest.segment_frame = original_segment_frame

    assert result["dataset"].handler is DummyDataset.handler
    assert result["test_df"] == "test_df"
    assert result["feature_columns"] == ["rsi_6", "vix_level"]
    assert calls == {
        "feature_packs": ("vix_term", "transition_signals"),
        "profile_name": "custom_profile",
        "feature_columns_packs": ("vix_term", "transition_signals"),
        "feature_columns_profile_name": "custom_profile",
    }


def test_persist_actor_bundle_creates_profile_artifact_directory():
    class DummyActor:
        output_dim = 3

        def state_dict(self):
            return {}

    profile = load_profile("smoke")
    with TemporaryDirectory() as tmpdir:
        previous_settings_model_dir = settings.MODEL_DIR
        previous_artifacts_model_dir = artifacts.MODEL_DIR
        settings.MODEL_DIR = Path(tmpdir)
        artifacts.MODEL_DIR = Path(tmpdir)
        try:
            path = artifacts.persist_actor_bundle(
                profile=profile,
                actor=DummyActor(),
                obs_dim=1,
                seed=1,
                effective_tag="",
            )
        finally:
            settings.MODEL_DIR = previous_settings_model_dir
            artifacts.MODEL_DIR = previous_artifacts_model_dir

        assert path.name == "soxl_smoke_actor.pt"
        assert path.exists()


if __name__ == "__main__":
    test_default_artifact_paths_stay_canonical()
    test_namespaced_artifact_paths_do_not_clobber_main_profile_outputs()
    test_artifact_tag_is_sanitized_for_worker_names()
    test_walkforward_validator_uses_canonical_profile_model_path()
    test_export_profile_daily_backtest_resolves_saved_walkforward_actor_layout()
    test_export_profile_daily_backtest_prefers_explicit_model_path()
    test_export_profile_daily_backtest_uses_profile_specific_feature_packs()
    test_persist_actor_bundle_creates_profile_artifact_directory()
    print("test_artifact_paths: OK")
