#!/usr/bin/env python3
"""Validate research-knowledge indexes and active workpad size budgets."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "docs" / "research_knowledge"
ACTIVE_DIR = ROOT / "codex_exec_plans" / "active"
REQUIRED_FILES = [
    "factors.md",
    "data.md",
    "model_selection.md",
    "networks.md",
    "rewards.md",
    "execution_overlays.md",
    "multi_asset.md",
]
REQUIRED_HEADINGS = [
    "## Retained Insight",
    "## Failed Pattern",
    "## Evidence Path",
    "## Reusable Method",
    "## Next Valid Experiment Type",
]
ACTIVE_LINE_BUDGET = 300


def main() -> int:
    failures: list[str] = []
    index_path = KNOWLEDGE_DIR / "index.md"
    index_text = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
    if not index_text:
        failures.append("missing docs/research_knowledge/index.md")
    for name in REQUIRED_FILES:
        path = KNOWLEDGE_DIR / name
        if not path.exists():
            failures.append(f"missing {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for heading in REQUIRED_HEADINGS:
            if heading not in text:
                failures.append(f"{path.relative_to(ROOT)} missing heading {heading}")
        if name not in index_text:
            failures.append(f"index.md does not link {name}")
    for path in ACTIVE_DIR.glob("*.md"):
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > ACTIVE_LINE_BUDGET:
            failures.append(f"{path.relative_to(ROOT)} has {line_count} lines > {ACTIVE_LINE_BUDGET}")
    if failures:
        print("check_research_knowledge: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(
        "check_research_knowledge: OK "
        f"files={len(REQUIRED_FILES)} active_budget={ACTIVE_LINE_BUDGET}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
