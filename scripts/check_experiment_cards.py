#!/usr/bin/env python3
"""Validate experiment-card lifecycle consistency."""
from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
REGISTRY = ROOT / "configs" / "soxl" / "profile_registry.yaml"
ALLOWED_STATUSES = {"canonical", "candidate", "archive"}


def registry_by_name() -> dict[str, dict]:
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {}
    return {str(item["name"]): item for item in payload.get("profiles", [])}


def package_dirs() -> list[Path]:
    dirs: list[Path] = []
    for status_dir in sorted(EXPERIMENTS.iterdir()):
        if not status_dir.is_dir() or status_dir.name not in ALLOWED_STATUSES:
            continue
        dirs.extend(sorted(path for path in status_dir.iterdir() if path.is_dir()))
    return dirs


def main() -> int:
    registry = registry_by_name()
    failures: list[str] = []
    rows: list[tuple[str, str, str, str, str]] = []
    for package in package_dirs():
        status_dir = package.parent.name
        name = package.name
        rel = package.relative_to(ROOT)
        if status_dir == "archive" and not (package / "profile.yaml").exists():
            continue
        if name not in registry:
            failures.append(f"{rel} has no profile_registry row")
            continue
        if status_dir in {"canonical", "candidate"} and not (package / "profile.yaml").exists():
            failures.append(f"{rel} missing profile.yaml")
        if status_dir in {"canonical", "candidate"} and not (package / "notes.md").exists():
            failures.append(f"{rel} missing notes.md")
        if status_dir in {"canonical", "candidate"} and not (package / "expectations.md").exists():
            failures.append(f"{rel} missing expectations.md")
        row = registry[name]
        rows.append(
            (
                name,
                status_dir,
                str(row.get("status", "")),
                str(row.get("family", "")),
                str(row.get("notes_ref", "")),
            )
        )
    if failures:
        print("check_experiment_cards: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"check_experiment_cards: OK packages={len(rows)} registry_rows={len(registry)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
