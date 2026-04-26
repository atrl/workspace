#!/usr/bin/env python3
"""Check skills for stale path references and outdated content."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / ".codex" / "skills"

# Known retired paths that should not appear in skills
RETIRED_PATHS = [
    "artifacts/models/",
    "artifacts/",
    "runtime/*/cards/",
    "runtime/soxl/post_train_review/",
    "memory/",
    "configs/soxl/",
]


def check_skill(skill_path: Path) -> list[str]:
    issues: list[str] = []
    try:
        text = skill_path.read_text(encoding="utf-8")
    except Exception as e:
        issues.append(f"  unreadable: {e}")
        return issues

    for retired in RETIRED_PATHS:
        if retired in text:
            issues.append(f"  stale path '{retired}'")
    return issues


def main() -> int:
    if not SKILLS_DIR.exists():
        print("[check_skills] skills dir missing")
        return 1

    all_ok = True
    for skill_md in sorted(SKILLS_DIR.rglob("SKILL.md")):
        issues = check_skill(skill_md)
        if issues:
            all_ok = False
            print(f"[check_skills] {skill_md.relative_to(ROOT)}")
            for issue in issues:
                print(issue)

    if all_ok:
        print("[check_skills] all skills clean")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
