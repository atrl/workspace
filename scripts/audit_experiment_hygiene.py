#!/usr/bin/env python3
"""Audit experiment hygiene: report mismatches, missing files, and stale artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
REGISTRY = ROOT / "configs" / "soxl" / "profile_registry.yaml"
MODEL_DIR = ROOT / "artifacts" / "models"
CONFIGS_DIR = ROOT / "configs" / "soxl"
ALLOWED_STATUSES = {"canonical", "candidate", "archive", "ablation", "dropped"}


def load_registry() -> dict[str, Any]:
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {}
    return {str(item["name"]): item for item in payload.get("profiles", [])}


def package_dirs() -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for status_dir in sorted(EXPERIMENTS.iterdir()):
        if not status_dir.is_dir():
            continue
        result[status_dir.name] = sorted(
            path for path in status_dir.iterdir() if path.is_dir()
        )
    return result


def count_lines(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except Exception:
        return 0


def profile_name_from_artifact(name: str) -> str | None:
    if not name.startswith("soxl_"):
        return None
    stem = name.removeprefix("soxl_")
    for suffix in ("_actor.pt", "_summary.json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit experiment hygiene.")
    parser.add_argument("--fix-suggestions", action="store_true", help="print fix commands")
    args = parser.parse_args()

    registry = load_registry()
    packages = package_dirs()
    issues: list[str] = []
    suggestions: list[str] = []

    # 1. Candidate packages with wrong registry status
    for pkg in packages.get("candidate", []):
        name = pkg.name
        row = registry.get(name)
        if row and row.get("status") not in {"candidate", "canonical"}:
            issues.append(
                f"[DIR_MISMATCH] {pkg.relative_to(ROOT)}: registry status='{row.get('status')}'"
            )
            suggestions.append(f"mv {pkg.relative_to(ROOT)} experiments/{row.get('status')}/{name}")

    # 2. Archive packages with long notes
    for pkg in packages.get("archive", []):
        notes = pkg / "notes.md"
        if notes.exists() and count_lines(notes) > 50:
            issues.append(
                f"[LONG_NOTES] {pkg.relative_to(ROOT)}/notes.md has {count_lines(notes)} lines (limit 50)"
            )

    # 3. Missing expectations.md in candidate/canonical
    for status in ("candidate", "canonical"):
        for pkg in packages.get(status, []):
            if not (pkg / "expectations.md").exists():
                issues.append(
                    f"[MISSING_EXPECTATIONS] {pkg.relative_to(ROOT)} lacks expectations.md"
                )

    # 4. Legacy yaml files without experiments package
    for yaml_path in sorted(CONFIGS_DIR.glob("*.yaml")):
        if yaml_path.name == "profile_registry.yaml":
            continue
        name = yaml_path.stem
        has_package = any(
            (pkg / "profile.yaml").exists() and pkg.name == name
            for status_pkgs in packages.values()
            for pkg in status_pkgs
        )
        if not has_package:
            row = registry.get(name)
            status = row.get("status", "unknown") if row else "unknown"
            issues.append(
                f"[LEGACY_YAML] {yaml_path.relative_to(ROOT)} has no experiments/ package (registry={status})"
            )

    # 5. Artifacts that should be pruned
    for artifact in sorted(MODEL_DIR.glob("*")):
        name = artifact.name
        profile_name = profile_name_from_artifact(name)
        if not profile_name:
            continue
        row = registry.get(profile_name)
        if name.endswith("_actor.pt"):
            if row is None:
                issues.append(f"[ORPHAN_ACTOR] {artifact.relative_to(ROOT)}: no registry profile")
            elif not row.get("keep_actor", False):
                issues.append(
                    f"[PRUNE_ACTOR] {artifact.relative_to(ROOT)}: registry keep_actor=false"
                )
                suggestions.append(f"rm {artifact.relative_to(ROOT)}")
        elif name.endswith("_summary.json"):
            if row is not None and not row.get("keep_summary", True):
                issues.append(
                    f"[PRUNE_SUMMARY] {artifact.relative_to(ROOT)}: registry keep_summary=false"
                )
                suggestions.append(f"rm {artifact.relative_to(ROOT)}")

    if not issues:
        print("audit_experiment_hygiene: OK")
        return 0

    print("audit_experiment_hygiene: ISSUES FOUND")
    for issue in issues:
        print(f"  {issue}")
    if args.fix_suggestions and suggestions:
        print("\nSuggested fixes:")
        for s in suggestions:
            print(f"  {s}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
