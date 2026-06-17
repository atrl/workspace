#!/usr/bin/env python3
from __future__ import annotations

"""Repo-owned validation entrypoint for the SOXL harness."""

import argparse
import importlib.util
import py_compile
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "tests"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TEST_FILES = [
    "tests/test_artifact_paths.py",
    "tests/test_barrier_labels.py",
    "tests/test_checkpoint_review_assets.py",
    "tests/test_checkpoint_retention.py",
    "tests/test_drawdown_budget_overlay.py",
    "tests/test_budget_counterfactual.py",
    "tests/test_budget_usage_runtime_promotion_review.py",
    "tests/test_budget_usage_bc.py",
    "tests/test_factor_evaluation.py",
    "tests/test_hierarchical_composer.py",
    "tests/test_position_cap.py",
    "tests/test_regime_features.py",
    "tests/test_runtime.py",
    "tests/test_two_speed_budget.py",
    "tests/test_update_market_data.py",
    "tests/test_profile_registry.py",
    "tests/test_production_schedule.py",
    "tests/test_reentry_labels.py",
    "tests/test_execution_overlay_registry.py",
    "tests/test_selective_release_overlay.py",
    "tests/test_selective_release_walkforward.py",
    "tests/test_soxx_relative_walkforward.py",
    "tests/test_selective_release_promotion_review.py",
    "tests/test_runtime_selection_review.py",
    "tests/test_model_runtime_readiness_review.py",
    "tests/test_metric_reconciliation.py",
    "tests/test_state_objective_bc.py",
    "tests/test_state_objective_training_targets.py",
    "tests/test_oos_diagnostics.py",
    "tests/test_participation_counterfactual.py",
    "tests/test_purged_walkforward.py",
    "tests/test_scheduler_health.py",
    "tests/test_simplified_architecture.py",
    "tests/test_vercel_dashboard.py",
]

DEFAULT_COMPILE_TARGETS = [
    "soxl_qlib_strategy_v2.py",
    "src/soxl_quant/settings.py",
    "src/soxl_quant/data.py",
    "src/soxl_quant/runtime.py",
    "src/soxl_quant/app/barrier_labels.py",
    "src/soxl_quant/app/checkpoint_retention.py",
    "src/soxl_quant/app/drawdown_budget_overlay.py",
    "src/soxl_quant/app/budget_counterfactual.py",
    "src/soxl_quant/app/oos_diagnostics.py",
    "src/soxl_quant/app/participation_counterfactual.py",
    "src/soxl_quant/app/purged_walkforward.py",
    "src/soxl_quant/app/reentry_labels.py",
    "src/soxl_quant/app/simplified_architecture.py",
    "src/soxl_quant/app/simplified_cli.py",
    "src/soxl_quant/app/state_objective_bc.py",
    "src/soxl_quant/app/state_objective_training_targets.py",
    "src/soxl_quant/core/metric_reconciliation.py",
    "src/soxl_quant/core/runtime_state.py",
    "src/soxl_quant/policy/execution_overlays.py",
    "src/soxl_quant/policy/hierarchical_composer.py",
    "src/soxl_quant/policy/mode_conditioning.py",
    "src/soxl_quant/policy/two_speed_budget.py",
    "src/soxl_quant/policy/usage_regression.py",
    "scripts/build_backtest_dashboard.py",
    "scripts/build_runtime_index.py",
    "scripts/check_dashboard_static_export.py",
    "scripts/check_skills.py",
    "scripts/check_experiment_cards.py",
    "scripts/check_harness.py",
    "scripts/check_research_knowledge.py",
    "scripts/check_research_task_router.py",
    "scripts/check_source_boundaries.py",
    "scripts/checkpoint_review.py",
    "scripts/compose_budget_usage_counterfactual.py",
    "scripts/deploy_vercel_dashboard.py",
    "scripts/evaluate_intraday_rejection.py",
    "scripts/evaluate_signed_te_reversion.py",
    "scripts/evaluate_signed_te_stress_gate.py",
    "scripts/export_profile_daily_backtest.py",
    "scripts/evaluate_selective_release_overlay.py",
    "scripts/evaluate_selective_release_walkforward.py",
    "scripts/evaluate_soxx_relative_overlay.py",
    "scripts/evaluate_soxx_relative_walkforward.py",
    "scripts/diagnose_oos_poor_windows.py",
    "scripts/review_selective_release_promotion.py",
    "scripts/review_runtime_selection.py",
    "scripts/review_budget_usage_runtime_promotion.py",
    "scripts/review_model_runtime_readiness.py",

    "scripts/prune_artifacts.py",
    "scripts/rank_counterfactual_variants.py",

    "scripts/update_data_registry.py",
    "production/composed_session.py",
    "production/soxl_daily_scheduler.py",
    "production/soxl_paper_trading.py",
    "production/soxl_paper_runner.py",
]

DEFAULT_COMMAND_CHECKS = [
    ["scripts/build_runtime_index.py", "--check"],
    ["scripts/check_research_knowledge.py"],
    ["scripts/check_research_task_router.py"],
    ["scripts/check_simplified_architecture.py"],
    ["scripts/check_source_boundaries.py"],
    ["scripts/check_skills.py"],
    ["scripts/check_experiment_cards.py"],
    ["scripts/update_data_registry.py", "--check"],
]


def load_module(path: Path):
    module_name = "repo_check_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_test_file(path: Path) -> tuple[int, list[str]]:
    module = load_module(path)
    failures: list[str] = []
    passed = 0
    for name in sorted(dir(module)):
        if not name.startswith("test_"):
            continue
        candidate = getattr(module, name)
        if not callable(candidate):
            continue
        try:
            candidate()
            passed += 1
        except Exception:
            failures.append(
                f"{path}:{name}\n{traceback.format_exc().rstrip()}\n"
            )
    return passed, failures


def run_compile(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{path}\n{exc.msg}")
    return failures


def run_command_checks(commands: list[list[str]]) -> list[str]:
    failures: list[str] = []
    for command in commands:
        proc = subprocess.run(
            [sys.executable, *command],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        label = " ".join(command)
        if proc.returncode == 0:
            print(f"  - {label}: ok")
            continue
        failures.append(
            f"{label}\nstdout:\n{proc.stdout.rstrip()}\nstderr:\n{proc.stderr.rstrip()}"
        )
        print(f"  - {label}: failed")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repo-owned harness checks.")
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="skip py_compile checks",
    )
    parser.add_argument(
        "--test-file",
        action="append",
        default=[],
        help="additional test file to run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_paths = [ROOT / path for path in DEFAULT_TEST_FILES + args.test_file]
    compile_paths = [ROOT / path for path in DEFAULT_COMPILE_TARGETS]

    total_passed = 0
    failures: list[str] = []

    print("[check] running assertion modules")
    for path in test_paths:
        try:
            passed, test_failures = run_test_file(path)
            total_passed += passed
            failures.extend(test_failures)
            status = "ok" if not test_failures else "failed"
        except Exception:
            passed = 0
            test_failures = [f"{path}:import\n{traceback.format_exc().rstrip()}\n"]
            failures.extend(test_failures)
            status = "failed"
        print(f"  - {path.relative_to(ROOT)}: {status} ({passed} checks)")

    compile_failures: list[str] = []
    if not args.skip_compile:
        print("[check] running py_compile")
        compile_failures = run_compile(compile_paths)
        failures.extend(compile_failures)
        if compile_failures:
            print(f"  - py_compile: failed ({len(compile_failures)} targets)")
        else:
            print(f"  - py_compile: ok ({len(compile_paths)} targets)")

    print("[check] running repository boundary checks")
    failures.extend(run_command_checks(DEFAULT_COMMAND_CHECKS))

    print(f"[summary] passed={total_passed} failures={len(failures)}")
    if failures:
        print("[failures]")
        for failure in failures:
            print(failure.rstrip())
            print("---")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
