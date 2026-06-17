from __future__ import annotations

from src.soxl_quant.app.simplified_architecture import (
    FORBIDDEN_SIMPLIFIED_IMPORT_PATTERNS,
    PRIMARY_SIMPLIFIED_ARTIFACT,
    SIMPLIFIED_NON_RL_SOURCE_PATHS,
    check_simplified_non_rl_boundaries,
    simplified_architecture_manifest,
)


def test_simplified_architecture_manifest_points_to_alpha158_supervised_cadence_surface():
    manifest = simplified_architecture_manifest()

    assert manifest["primary_surface"] == "qlib_alpha158_supervised_cadence_v1"
    assert manifest["primary_training_kind"] == "supervised"
    assert manifest["rl_training_primary"] is False
    assert manifest["primary_artifact"] == str(PRIMARY_SIMPLIFIED_ARTIFACT)
    assert manifest["primary_artifact_exists"] is True
    assert manifest["data"]["feature_packs"][0] == "qlib_alpha158"
    assert manifest["data"]["feature_count"] >= 158
    assert manifest["dynamic_trade_cadence"]["enabled"] is True
    assert manifest["dynamic_trade_cadence"]["cadence_pair_rows"] == 20
    assert manifest["dynamic_trade_cadence"]["test_2022_trade_reduction_rows"] == 20
    assert manifest["dynamic_trade_cadence"]["test_2022_turnover_reduction_rows"] == 20
    assert manifest["promotion"]["eligible_profiles"] == []


def test_simplified_non_rl_boundary_excludes_policy_and_qlib_rl_imports():
    violations = check_simplified_non_rl_boundaries()

    assert violations == []
    assert "src/soxl_quant/app/quantbench_supervised.py" in SIMPLIFIED_NON_RL_SOURCE_PATHS
    assert "src/soxl_quant/app/train.py" not in SIMPLIFIED_NON_RL_SOURCE_PATHS
    assert "qlib.rl" in FORBIDDEN_SIMPLIFIED_IMPORT_PATTERNS


def test_simplified_architecture_manifest_marks_remaining_rl_as_compatibility_only():
    manifest = simplified_architecture_manifest()

    assert manifest["source_boundary"]["violations"] == []
    assert manifest["primary_non_rl_surface_available"] is True
    assert manifest["goal_complete"] is False
    assert "src/soxl_quant/app/train.py" in manifest["remaining_rl_compatibility_surfaces"]
    assert "src/soxl_quant/policy/" in manifest["remaining_rl_compatibility_surfaces"]
