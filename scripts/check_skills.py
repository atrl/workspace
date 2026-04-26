#!/usr/bin/env python3
"""Check skills for stale path references and outdated content."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / ".codex" / "skills"

# Known retired path patterns that should not appear in skills
# Each tuple is (regex_pattern, human_readable_description)
RETIRED_PATTERNS: list[tuple[str, str]] = [
    (r"runtime/\w+/cards/", "runtime/<dir>/cards/"),
    (r"artifacts/models/", "artifacts/models/"),
    (r"artifacts/(?!retention_manifest)", "artifacts/"),  # allow retention_manifest mention if needed
    (r"runtime/soxl/post_train_review/", "runtime/soxl/post_train_review/"),
    (r"memory/", "memory/"),
    (r"configs/soxl/", "configs/soxl/"),
]


def check_skill(skill_path: Path) -> list[str]:
    issues: list[str] = []
    try:
        text = skill_path.read_text(encoding="utf-8")
    except Exception as e:
        issues.append(f"  unreadable: {e}")
        return issues

    for pattern, description in RETIRED_PATTERNS:
        if re.search(pattern, text):
            issues.append(f"  stale path '{description}'")
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
