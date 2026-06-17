from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

import src.soxl_quant as package
import src.soxl_quant.app as app_package
from src.soxl_quant.app.simplified_architecture import (
    FORBIDDEN_SIMPLIFIED_IMPORT_PATTERNS,
    PRIMARY_SIMPLIFIED_ARTIFACT,
    SIMPLIFIED_NON_RL_SOURCE_PATHS,
    check_simplified_non_rl_boundaries,
    simplified_architecture_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


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
    assert "src/soxl_quant/core/runtime_state.py" in SIMPLIFIED_NON_RL_SOURCE_PATHS
    assert "src/soxl_quant/app/checkpoint_retention.py" in SIMPLIFIED_NON_RL_SOURCE_PATHS
    assert "src/soxl_quant/app/quantbench_supervised.py" in SIMPLIFIED_NON_RL_SOURCE_PATHS
    assert "src/soxl_quant/app/train.py" not in SIMPLIFIED_NON_RL_SOURCE_PATHS
    assert "qlib.rl" in FORBIDDEN_SIMPLIFIED_IMPORT_PATTERNS


def test_simplified_architecture_manifest_has_no_remaining_rl_compatibility_surfaces():
    manifest = simplified_architecture_manifest()

    assert manifest["source_boundary"]["violations"] == []
    assert manifest["primary_non_rl_surface_available"] is True
    assert manifest["goal_complete"] is True
    assert manifest["remaining_rl_compatibility_surfaces"] == []


def test_default_app_module_cli_emits_simplified_architecture_manifest_json():
    proc = subprocess.run(
        [sys.executable, "-m", "src.soxl_quant.app", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["primary_surface"] == "qlib_alpha158_supervised_cadence_v1"
    assert payload["rl_training_primary"] is False


def test_package_main_dispatches_to_simplified_architecture_cli():
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
        status = package.main(["--json"])

    assert status == 0
    payload = json.loads(stdout.getvalue())
    assert payload["primary_training_kind"] == "supervised"
    assert payload["primary_non_rl_surface_available"] is True


def test_app_package_main_dispatches_to_simplified_architecture_cli():
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
        status = app_package.main(["--json"])

    assert status == 0
    payload = json.loads(stdout.getvalue())
    assert payload["default_entrypoints"]["legacy_rl_cli_primary"] is False


def test_default_entrypoints_do_not_reference_legacy_rl_cli():
    root_cli_source = (ROOT / "soxl_qlib_strategy_v2.py").read_text(encoding="utf-8")
    package_source = (ROOT / "src" / "soxl_quant" / "__init__.py").read_text(encoding="utf-8")
    app_package_source = (ROOT / "src" / "soxl_quant" / "app" / "__init__.py").read_text(encoding="utf-8")
    app_main_source = (ROOT / "src" / "soxl_quant" / "app" / "__main__.py").read_text(encoding="utf-8")

    assert "app.cli" not in root_cli_source
    assert "app.cli" not in package_source
    assert ".cli" not in app_package_source
    assert ".cli" not in app_main_source
    assert "reinforcement learning" not in root_cli_source.lower()
    assert "reinforcement learning" not in package_source.lower()


def test_simplified_manifest_records_default_entrypoint_migration():
    manifest = simplified_architecture_manifest()

    assert manifest["default_entrypoints"]["package_main"] == "src.soxl_quant.app.simplified_cli"
    assert manifest["default_entrypoints"]["app_package_main"] == "src.soxl_quant.app.simplified_cli"
    assert manifest["default_entrypoints"]["app_module_main"] == "src.soxl_quant.app.simplified_cli"
    assert manifest["default_entrypoints"]["legacy_rl_cli_primary"] is False
    assert manifest["default_entrypoints"]["violations"] == []
    assert "soxl_qlib_strategy_v2.py" in manifest["default_entrypoints"]["checked_paths"]


def test_legacy_rl_compatibility_facades_are_retired():
    assert not (ROOT / "src" / "soxl_quant" / "cli.py").exists()
    assert not (ROOT / "src" / "soxl_quant" / "rl.py").exists()

    manifest = simplified_architecture_manifest()
    assert "src/soxl_quant/cli.py" not in manifest["remaining_rl_compatibility_surfaces"]
    assert "src/soxl_quant/rl.py" not in manifest["remaining_rl_compatibility_surfaces"]


def test_legacy_app_rl_cli_entrypoint_is_retired():
    assert not (ROOT / "src" / "soxl_quant" / "app" / "cli.py").exists()

    manifest = simplified_architecture_manifest()
    assert "src/soxl_quant/app/cli.py" not in manifest["remaining_rl_compatibility_surfaces"]


def test_legacy_app_rl_training_entrypoints_are_retired():
    assert not (ROOT / "src" / "soxl_quant" / "app" / "train.py").exists()
    assert not (ROOT / "src" / "soxl_quant" / "app" / "workers.py").exists()
    assert not (ROOT / "src" / "soxl_quant" / "app" / "selection.py").exists()

    manifest = simplified_architecture_manifest()
    assert "src/soxl_quant/app/train.py" not in manifest["remaining_rl_compatibility_surfaces"]
    assert "src/soxl_quant/app/workers.py" not in manifest["remaining_rl_compatibility_surfaces"]
    assert "src/soxl_quant/app/selection.py" not in manifest["remaining_rl_compatibility_surfaces"]


def test_legacy_rl_reward_module_is_retired():
    assert not (ROOT / "src" / "soxl_quant" / "policy" / "reward.py").exists()
    assert not (ROOT / "tests" / "test_reward.py").exists()


def test_non_rl_runtime_state_helpers_are_outside_rl_environment():
    helper_path = ROOT / "src" / "soxl_quant" / "core" / "runtime_state.py"
    assert helper_path.exists()

    helper_source = helper_path.read_text(encoding="utf-8")
    for forbidden in ("qlib.rl", "gymnasium", "tianshou", "policy.environment"):
        assert forbidden not in helper_source

    migrated_paths = [
        "src/soxl_quant/runtime.py",
        "src/soxl_quant/backtest.py",
        "src/soxl_quant/policy/observation.py",
        "src/soxl_quant/policy/usage_regression.py",
        "tests/test_runtime.py",
    ]
    offenders = [
        rel_path
        for rel_path in migrated_paths
        if "policy.environment" in (ROOT / rel_path).read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []


def test_legacy_rl_environment_harness_coverage_is_retired():
    assert not (ROOT / "tests" / "test_structural_model.py").exists()

    check_harness_source = (ROOT / "scripts" / "check_harness.py").read_text(encoding="utf-8")
    assert '"tests/test_structural_model.py"' not in check_harness_source

    position_cap_source = (ROOT / "tests" / "test_position_cap.py").read_text(encoding="utf-8")
    assert "policy.environment" not in position_cap_source
    assert "PositionAction" not in position_cap_source


def test_runtime_no_longer_imports_actor_builder_at_module_import_time():
    runtime_source = (ROOT / "src" / "soxl_quant" / "runtime.py").read_text(encoding="utf-8")

    forbidden = (
        "policy.policy_builder",
        "actor_action",
        "build_policy",
        "policy.observation",
        "target_position_for_profile",
    )
    offenders = [pattern for pattern in forbidden if pattern in runtime_source]
    assert offenders == []


def test_backtest_module_no_longer_imports_actor_builder_at_module_import_time():
    backtest_source = (ROOT / "src" / "soxl_quant" / "backtest.py").read_text(encoding="utf-8")

    forbidden = (
        "policy.policy_builder",
        "actor_action",
        "tianshou.utils.net.discrete",
    )
    offenders = [pattern for pattern in forbidden if pattern in backtest_source]
    assert offenders == []


def test_harness_no_longer_keeps_actor_builder_consumers_alive():
    assert not (ROOT / "tests" / "test_budget_usage_bc.py").exists()
    assert not (ROOT / "tests" / "test_state_objective_bc.py").exists()
    assert not (ROOT / "src" / "soxl_quant" / "app" / "state_objective_bc.py").exists()

    check_harness_source = (ROOT / "scripts" / "check_harness.py").read_text(encoding="utf-8")
    forbidden = (
        "tests/test_budget_usage_bc.py",
        "tests/test_state_objective_bc.py",
        "src/soxl_quant/app/state_objective_bc.py",
        "scripts/export_profile_daily_backtest.py",
        "scripts/evaluate_selective_release_overlay.py",
        "scripts/evaluate_selective_release_walkforward.py",
        "scripts/evaluate_soxx_relative_overlay.py",
    )
    offenders = [pattern for pattern in forbidden if pattern in check_harness_source]
    assert offenders == []


def test_legacy_rl_actor_builder_modules_are_retired():
    retired_paths = [
        "production/composed_session.py",
        "production/futu_manual_session.py",
        "production/ib_manual_session.py",
        "production/run_soxl_weekly_wf.sh",
        "production/soxl_futu_runner.py",
        "production/soxl_paper_runner.py",
        "src/soxl_quant/compat.py",
        "src/soxl_quant/app/bootstrap.py",
        "src/soxl_quant/policy/environment.py",
        "src/soxl_quant/policy/evaluation.py",
        "src/soxl_quant/policy/policy_builder.py",
        "src/soxl_quant/policy/models/attention.py",
        "src/soxl_quant/policy/models/gru.py",
        "src/soxl_quant/policy/models/mlp.py",
        "src/soxl_quant/policy/models/moe.py",
        "src/soxl_quant/policy/models/temporal_conv.py",
        "src/soxl_quant/policy/models/tide.py",
        "tests/test_policy_networks.py",
    ]
    offenders = [rel_path for rel_path in retired_paths if (ROOT / rel_path).exists()]
    assert offenders == []


def test_legacy_runtime_cards_do_not_import_retired_app_rl_cli():
    runtime_card_scripts = sorted((ROOT / "runtime").glob("**/*.py"))
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in runtime_card_scripts
        if "src.soxl_quant.app.cli" in path.read_text(encoding="utf-8", errors="ignore")
    ]

    assert offenders == []


def test_scripts_do_not_import_retired_app_training_surfaces():
    scripts = sorted((ROOT / "scripts").glob("*.py"))
    forbidden = (
        "src.soxl_quant.app.selection",
        "src.soxl_quant.app.train",
        "src.soxl_quant.app.workers",
    )
    offenders = [
        f"{path.relative_to(ROOT).as_posix()}:{pattern}"
        for path in scripts
        for pattern in forbidden
        if pattern in path.read_text(encoding="utf-8", errors="ignore")
    ]

    assert offenders == []


def test_harness_compile_targets_do_not_keep_retired_rl_facades_alive():
    check_harness_source = (ROOT / "scripts" / "check_harness.py").read_text(encoding="utf-8")

    assert '"src/soxl_quant/cli.py"' not in check_harness_source
    assert '"src/soxl_quant/rl.py"' not in check_harness_source
    assert '"src/soxl_quant/app/cli.py"' not in check_harness_source
    assert '"src/soxl_quant/app/selection.py"' not in check_harness_source
    assert '"src/soxl_quant/app/train.py"' not in check_harness_source
    assert '"src/soxl_quant/app/workers.py"' not in check_harness_source
    assert '"tests/test_train_summary_persistence.py"' not in check_harness_source
    assert '"tests/test_reward.py"' not in check_harness_source
    assert '"src/soxl_quant/core/runtime_state.py"' in check_harness_source


def test_harness_runs_simplified_architecture_guard_by_default():
    check_harness_source = (ROOT / "scripts" / "check_harness.py").read_text(encoding="utf-8")

    assert '"tests/test_simplified_architecture.py"' in check_harness_source
    assert '"src/soxl_quant/app/simplified_architecture.py"' in check_harness_source
    assert '"src/soxl_quant/app/simplified_cli.py"' in check_harness_source
    assert '["scripts/check_simplified_architecture.py"]' in check_harness_source
