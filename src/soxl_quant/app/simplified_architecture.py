from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from ..settings import ROOT

PRIMARY_SIMPLIFIED_SURFACE = "qlib_alpha158_supervised_cadence_v1"
PRIMARY_SIMPLIFIED_ARTIFACT = (
    Path("runtime")
    / "model_selection"
    / "card"
    / PRIMARY_SIMPLIFIED_SURFACE
    / "summary.json"
)
PRIMARY_SUPERVISED_SUMMARY = PRIMARY_SIMPLIFIED_ARTIFACT.with_name("supervised_summary.json")
PRIMARY_CADENCE_METRICS = PRIMARY_SIMPLIFIED_ARTIFACT.with_name("cadence_pair_metrics.csv")

SIMPLIFIED_NON_RL_SOURCE_PATHS: tuple[str, ...] = (
    "src/soxl_quant/data.py",
    "src/soxl_quant/app/quantbench_objectives.py",
    "src/soxl_quant/app/quantbench_screen.py",
    "src/soxl_quant/app/quantbench_supervised.py",
)
FORBIDDEN_SIMPLIFIED_IMPORT_PATTERNS: tuple[str, ...] = (
    "qlib.rl",
    "src.soxl_quant.policy",
    "soxl_quant.policy",
    "src.soxl_quant.rl",
    "soxl_quant.rl",
)
DEFAULT_ENTRYPOINT_SOURCE_PATHS: tuple[str, ...] = (
    "soxl_qlib_strategy_v2.py",
    "src/soxl_quant/__init__.py",
    "src/soxl_quant/app/__init__.py",
    "src/soxl_quant/app/__main__.py",
)
FORBIDDEN_DEFAULT_ENTRYPOINT_PATTERNS: tuple[str, ...] = (
    "app.cli",
    "from .cli",
    "run_profile",
    "PPO",
    "reinforcement learning",
)
REMAINING_RL_COMPATIBILITY_SURFACES: tuple[str, ...] = (
    "src/soxl_quant/app/train.py",
    "src/soxl_quant/app/selection.py",
    "src/soxl_quant/app/workers.py",
    "src/soxl_quant/policy/",
    "production/",
)


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def _count_truthy_csv_rows(path: Path, column: str) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for row in reader if str(row.get(column, "")).strip().lower() == "true")


def check_simplified_non_rl_boundaries(root: Path | None = None) -> list[dict[str, str]]:
    """Return forbidden RL import references inside the simplified supervised path."""
    repo_root = Path(root or ROOT)
    violations: list[dict[str, str]] = []
    for rel_path in SIMPLIFIED_NON_RL_SOURCE_PATHS:
        path = repo_root / rel_path
        if not path.exists():
            violations.append({"path": rel_path, "pattern": "missing", "line": "0"})
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            for pattern in FORBIDDEN_SIMPLIFIED_IMPORT_PATTERNS:
                if pattern in line:
                    violations.append({"path": rel_path, "pattern": pattern, "line": str(line_no)})
    return violations


def check_default_entrypoint_boundaries(root: Path | None = None) -> list[dict[str, str]]:
    """Return legacy RL references from package default entrypoints."""
    repo_root = Path(root or ROOT)
    violations: list[dict[str, str]] = []
    for rel_path in DEFAULT_ENTRYPOINT_SOURCE_PATHS:
        path = repo_root / rel_path
        if not path.exists():
            violations.append({"path": rel_path, "pattern": "missing", "line": "0"})
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            for pattern in FORBIDDEN_DEFAULT_ENTRYPOINT_PATTERNS:
                if pattern.lower() in line.lower():
                    violations.append({"path": rel_path, "pattern": pattern, "line": str(line_no)})
    return violations


def simplified_architecture_manifest(root: Path | None = None) -> dict[str, Any]:
    """Build the current non-RL simplification manifest from runtime evidence."""
    repo_root = Path(root or ROOT)
    primary_path = repo_root / PRIMARY_SIMPLIFIED_ARTIFACT
    supervised_summary_path = repo_root / PRIMARY_SUPERVISED_SUMMARY
    cadence_metrics_path = repo_root / PRIMARY_CADENCE_METRICS

    primary_summary: dict[str, Any] = {}
    if primary_path.exists():
        loaded = _load_json(primary_path)
        primary_summary = loaded if isinstance(loaded, dict) else {}

    supervised_summary: dict[str, Any] = {}
    if supervised_summary_path.exists():
        loaded = _load_json(supervised_summary_path)
        supervised_summary = loaded if isinstance(loaded, dict) else {}

    supervised_data = supervised_summary.get("data") if isinstance(supervised_summary.get("data"), dict) else {}
    screen = supervised_summary.get("screen") if isinstance(supervised_summary.get("screen"), dict) else {}
    cadence_pair_rows = int(primary_summary.get("cadence_pair_rows") or _count_csv_rows(cadence_metrics_path))
    trade_reduction_rows = int(
        primary_summary.get("test_2022_trade_reduction_rows")
        or _count_truthy_csv_rows(cadence_metrics_path, "reduced_test_2022_trades")
    )
    turnover_reduction_rows = int(
        primary_summary.get("test_2022_turnover_reduction_rows")
        or _count_truthy_csv_rows(cadence_metrics_path, "reduced_test_2022_turnover")
    )
    feature_packs = [str(item) for item in _as_list(supervised_data.get("feature_packs"))]
    boundary_violations = check_simplified_non_rl_boundaries(repo_root)
    entrypoint_violations = check_default_entrypoint_boundaries(repo_root)

    primary_available = bool(
        primary_path.exists()
        and supervised_summary_path.exists()
        and not boundary_violations
        and not entrypoint_violations
        and feature_packs
        and feature_packs[0] == "qlib_alpha158"
        and int(supervised_data.get("feature_count") or 0) >= 158
        and cadence_pair_rows > 0
        and trade_reduction_rows > 0
        and turnover_reduction_rows > 0
    )

    remaining_rl_surfaces = [
        rel_path
        for rel_path in REMAINING_RL_COMPATIBILITY_SURFACES
        if (repo_root / rel_path).exists()
    ]

    return {
        "primary_surface": PRIMARY_SIMPLIFIED_SURFACE,
        "primary_training_kind": "supervised",
        "rl_training_primary": False,
        "primary_artifact": str(PRIMARY_SIMPLIFIED_ARTIFACT),
        "primary_artifact_exists": primary_path.exists(),
        "supervised_summary": str(PRIMARY_SUPERVISED_SUMMARY),
        "supervised_summary_exists": supervised_summary_path.exists(),
        "data": {
            "start": supervised_data.get("start"),
            "end": supervised_data.get("end"),
            "rows": supervised_data.get("rows"),
            "feature_count": supervised_data.get("feature_count"),
            "feature_packs": feature_packs,
            "model_names": [str(item) for item in _as_list(supervised_data.get("model_names"))],
            "train_window": _as_list(supervised_data.get("train_window")),
            "test_window": _as_list(supervised_data.get("test_window")),
        },
        "dynamic_trade_cadence": {
            "enabled": cadence_pair_rows > 0,
            "cadence_metrics": str(PRIMARY_CADENCE_METRICS),
            "cadence_pair_rows": cadence_pair_rows,
            "test_2022_trade_reduction_rows": trade_reduction_rows,
            "test_2022_turnover_reduction_rows": turnover_reduction_rows,
            "trade_cadences": [str(item) for item in _as_list(supervised_data.get("trade_cadences"))],
        },
        "default_entrypoints": {
            "package_main": "src.soxl_quant.app.simplified_cli",
            "app_package_main": "src.soxl_quant.app.simplified_cli",
            "app_module_main": "src.soxl_quant.app.simplified_cli",
            "legacy_rl_cli_primary": False,
            "checked_paths": list(DEFAULT_ENTRYPOINT_SOURCE_PATHS),
            "forbidden_patterns": list(FORBIDDEN_DEFAULT_ENTRYPOINT_PATTERNS),
            "violations": entrypoint_violations,
        },
        "promotion": {
            "decision": primary_summary.get("decision") or supervised_summary.get("decision"),
            "eligible_profiles": _as_list(primary_summary.get("eligible_profiles"))
            or _as_list(screen.get("eligible_profiles")),
            "promotion_allowed": bool(primary_summary.get("promotion_allowed", False)),
        },
        "source_boundary": {
            "checked_paths": list(SIMPLIFIED_NON_RL_SOURCE_PATHS),
            "forbidden_patterns": list(FORBIDDEN_SIMPLIFIED_IMPORT_PATTERNS),
            "violations": boundary_violations,
        },
        "primary_non_rl_surface_available": primary_available,
        "remaining_rl_compatibility_surfaces": remaining_rl_surfaces,
        "goal_complete": primary_available and not remaining_rl_surfaces,
        "next_action": (
            "Keep the Alpha158 supervised/cadence path as primary evidence, then migrate entrypoints "
            "and production runtime dependencies before deleting legacy RL compatibility surfaces."
        ),
    }


def simplified_architecture_failures(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not manifest.get("primary_artifact_exists"):
        failures.append(f"missing primary artifact: {manifest.get('primary_artifact')}")
    if not manifest.get("supervised_summary_exists"):
        failures.append(f"missing supervised summary: {manifest.get('supervised_summary')}")
    if manifest.get("source_boundary", {}).get("violations"):
        failures.append("simplified source boundary has RL import violations")
    if manifest.get("default_entrypoints", {}).get("violations"):
        failures.append("default package entrypoints still reference legacy RL CLI")
    if not manifest.get("data", {}).get("feature_packs") or manifest["data"]["feature_packs"][0] != "qlib_alpha158":
        failures.append("primary supervised feature pack is not qlib_alpha158")
    if int(manifest.get("data", {}).get("feature_count") or 0) < 158:
        failures.append("primary supervised feature count does not cover Alpha158")
    if int(manifest.get("dynamic_trade_cadence", {}).get("cadence_pair_rows") or 0) <= 0:
        failures.append("dynamic trade cadence evidence is missing")
    if int(manifest.get("dynamic_trade_cadence", {}).get("test_2022_trade_reduction_rows") or 0) <= 0:
        failures.append("dynamic cadence did not reduce test_2022 trades in evidence")
    if int(manifest.get("dynamic_trade_cadence", {}).get("test_2022_turnover_reduction_rows") or 0) <= 0:
        failures.append("dynamic cadence did not reduce test_2022 turnover in evidence")
    return failures


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the simplified non-RL SOXL architecture surface.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true", help="Print the full manifest as JSON.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest = simplified_architecture_manifest(args.root)
    failures = simplified_architecture_failures(manifest)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif failures:
        print("check_simplified_architecture: FAIL")
        for failure in failures:
            print(f"- {failure}")
    else:
        print(
            "check_simplified_architecture: OK "
            f"primary_surface={manifest['primary_surface']} "
            f"cadence_pairs={manifest['dynamic_trade_cadence']['cadence_pair_rows']} "
            f"remaining_rl_compatibility_surfaces={len(manifest['remaining_rl_compatibility_surfaces'])}"
        )
    return 1 if failures else 0


__all__ = [
    "DEFAULT_ENTRYPOINT_SOURCE_PATHS",
    "FORBIDDEN_SIMPLIFIED_IMPORT_PATTERNS",
    "PRIMARY_SIMPLIFIED_ARTIFACT",
    "PRIMARY_SIMPLIFIED_SURFACE",
    "SIMPLIFIED_NON_RL_SOURCE_PATHS",
    "check_default_entrypoint_boundaries",
    "check_simplified_non_rl_boundaries",
    "main",
    "simplified_architecture_failures",
    "simplified_architecture_manifest",
]
