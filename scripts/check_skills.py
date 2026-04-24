#!/usr/bin/env python3
"""
Skill health checker:
1. Verify all skills are indexed in .codex/skills/_index.json
2. Check for hard-coded 'SOXL' in skill content (should use generic terms)
3. Flag stale skills (unused beyond stale_after_days)
4. Verify SKILL.md version matches _index.json version
"""
import json, sys, re
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / ".codex" / "skills"
INDEX_PATH = SKILLS_DIR / "_index.json"


def load_index():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def check_index_completeness(index):
    """Ensure every skill dir is in the index."""
    indexed = set(index.get("skills", {}).keys())
    on_disk = set()
    for d in SKILLS_DIR.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            on_disk.add(d.name)

    missing_in_index = on_disk - indexed
    missing_on_disk = indexed - on_disk
    return missing_in_index, missing_on_disk


def check_soxl_hardcoding(skill_name):
    """Check if skill content contains hard-coded 'SOXL' references."""
    skill_md = SKILLS_DIR / skill_name / "SKILL.md"
    if not skill_md.exists():
        return []

    text = skill_md.read_text(encoding="utf-8")
    # Allow 'SOXL' in examples, aliases, or explicit multi-ETF context
    # Flag only if it appears to be the primary/only symbol without generic fallback
    issues = []
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        # Skip yaml front-matter aliases
        if line.strip().startswith("aliases:") or line.strip().startswith("-") and "soxl" in line.lower():
            continue
        # Skip lines that already mention multi-ETF support
        if any(k in line.lower() for k in ["3x", "tqqq", "upro", "spxl", "tecl", "leveraged", "标的"]):
            continue
        if re.search(r'\bSOXL\b', line):
            issues.append((i, line.strip()))
    return issues


def check_stale(index):
    """Flag skills unused beyond their stale_after_days."""
    stale = []
    today = datetime.now()
    for name, meta in index.get("skills", {}).items():
        last_used = meta.get("last_used", "")
        days = meta.get("stale_after_days", 30)
        try:
            lu = datetime.strptime(last_used, "%Y-%m-%d")
            if (today - lu).days > days:
                stale.append((name, (today - lu).days, days))
        except ValueError:
            pass
    return stale


def check_versions(index):
    """Verify SKILL.md version matches _index.json version."""
    mismatches = []
    for name, meta in index.get("skills", {}).items():
        expected = meta.get("version", "")
        skill_md = SKILLS_DIR / name / "SKILL.md"
        if not skill_md.exists():
            continue
        text = skill_md.read_text(encoding="utf-8")
        m = re.search(r'^version:\s*"?([^"\n]+)"?', text, re.MULTILINE)
        actual = m.group(1).strip() if m else ""
        if actual and actual != expected:
            mismatches.append((name, expected, actual))
    return mismatches


def main():
    if not INDEX_PATH.exists():
        print("ERROR: _index.json not found")
        sys.exit(1)

    index = load_index()
    errors = 0
    warnings = 0

    # 1. Completeness
    missing_in_index, missing_on_disk = check_index_completeness(index)
    if missing_in_index:
        print(f"ERROR: Skill dirs missing from index: {missing_in_index}")
        errors += 1
    if missing_on_disk:
        print(f"WARNING: Indexed skills missing on disk: {missing_on_disk}")
        warnings += 1

    # 2. SOXL hardcoding
    for name in index.get("skills", {}):
        issues = check_soxl_hardcoding(name)
        if issues:
            print(f"WARNING: {name} contains unqualified SOXL references:")
            for line_no, line in issues:
                print(f"  line {line_no}: {line[:80]}")
            warnings += 1

    # 3. Stale skills
    stale = check_stale(index)
    if stale:
        print("WARNING: Stale skills (unused beyond threshold):")
        for name, days, threshold in stale:
            print(f"  {name}: {days}d / threshold {threshold}d")
        warnings += 1

    # 4. Version mismatches
    mismatches = check_versions(index)
    if mismatches:
        print("ERROR: Version mismatches:")
        for name, expected, actual in mismatches:
            print(f"  {name}: index={expected}, SKILL.md={actual}")
        errors += 1

    print(f"\nSkill check complete: {errors} errors, {warnings} warnings.")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
