#!/usr/bin/env python3
"""Validate experiment-card lifecycle consistency."""
from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
REGISTRY = ROOT / "configs" / "soxl" / "profile_registry.yaml"
ALLOWED_STATUSES = {"canonical", "candidate", "archive", "ablation", "dropped", "archived"}

# Line-count limits per status (expectations merged into notes)
NOTES_MAX_LINES = {
    "canonical": 80,
    "candidate": 80,
    "archive": 50,
}


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


def count_lines(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except Exception:
        return 0


def main() -> int:
    registry = registry_by_name()
    failures: list[str] = []
    warnings: list[str] = []
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
        # expectations.md content merged into notes.md; no standalone file required

        # Check notes.md line count (optional)
        notes_path = package / "notes.md"
        if notes_path.exists():
            lines = count_lines(notes_path)
            limit = NOTES_MAX_LINES.get(status_dir, 60)
            if lines > limit:
                msg = f"{rel}/notes.md has {lines} lines (limit {limit})"
                if status_dir == "archive":
                    failures.append(msg)
                else:
                    warnings.append(msg)

        # Check directory status matches registry status
        row = registry[name]
        registry_status = str(row.get("status", ""))
        if registry_status not in ALLOWED_STATUSES:
            warnings.append(f"{rel}: unknown registry status '{registry_status}'")
        elif status_dir == "archive" and registry_status in {"archived", "ablation", "dropped"}:
            pass  # archive/ directory accepts any inactive status
        elif status_dir != registry_status:
            warnings.append(
                f"{rel}: directory status '{status_dir}' != registry status '{registry_status}'"
            )

        rows.append(
            (
                name,
                status_dir,
                registry_status,
                str(row.get("family", "")),
                str(row.get("notes_ref", "")),
            )
        )
    if failures:
        print("check_experiment_cards: FAIL")
        for failure in failures:
            print(f"- {failure}")
    if warnings:
        print("check_experiment_cards: WARNINGS")
        for warning in warnings:
            print(f"- {warning}")
    if not failures and not warnings:
        print(f"check_experiment_cards: OK packages={len(rows)} registry_rows={len(registry)}")
        return 0
    if not failures:
        print(f"check_experiment_cards: OK (with warnings) packages={len(rows)} registry_rows={len(registry)}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
