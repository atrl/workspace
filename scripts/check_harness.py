#!/usr/bin/env python3
from __future__ import annotations

"""Repo-owned validation entrypoint for the SOXL harness."""

import argparse
import importlib.util
import py_compile
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "tests"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TEST_FILES = [
    "tests/test_artifact_paths.py",
    "tests/test_checkpoint_review_assets.py",
    "tests/test_budget_usage_bc.py",
    "tests/test_hierarchical_composer.py",
    "tests/test_position_cap.py",
    "tests/test_regime_features.py",
    "tests/test_reward.py",
    "tests/test_runtime.py",
    "tests/test_structural_model.py",
    "tests/test_two_speed_budget.py",
]

DEFAULT_COMPILE_TARGETS = [
    "soxl_qlib_strategy_v2.py",
    "src/soxl_quant/settings.py",
    "src/soxl_quant/data.py",
    "src/soxl_quant/rl.py",
    "src/soxl_quant/runtime.py",
    "src/soxl_quant/policy/hierarchical_composer.py",
    "src/soxl_quant/policy/mode_conditioning.py",
    "src/soxl_quant/policy/two_speed_budget.py",
    "src/soxl_quant/policy/usage_regression.py",
    "src/soxl_quant/cli.py",
    "scripts/build_backtest_dashboard.py",
    "scripts/build_budget_usage_bc_actor.py",
    "scripts/build_budget_usage_oracle_actor.py",
    "scripts/build_budget_usage_regression_probe.py",
    "scripts/check_harness.py",
    "scripts/checkpoint_review.py",
    "scripts/hierarchical_online_budget_generalization.py",
    "scripts/memory_search.py",
    "scripts/memory_tools.py",
    "scripts/normalize_memory.py",
    "scripts/prune_artifacts.py",
    "scripts/rank_counterfactual_variants.py",
    "scripts/replay_two_speed_mode_budget.py",
    "production/soxl_paper_runner.py",
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
