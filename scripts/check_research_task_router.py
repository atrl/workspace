#!/usr/bin/env python3
from __future__ import annotations

"""Validate research task routing across knowledge docs, skills, memory queries, and runtime paths."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
ROUTER = ROOT / "docs" / "research_knowledge" / "task_router.json"
SKILL_DIR = ROOT / ".codex" / "skills"


def main() -> int:
    failures: list[str] = []
    if not ROUTER.exists():
        print("check_research_task_router: FAIL")
        print("- missing docs/research_knowledge/task_router.json")
        return 1
    payload = json.loads(ROUTER.read_text(encoding="utf-8"))
    directions = payload.get("directions", {})
    if not directions:
        failures.append("router has no directions")

    index_text = (ROOT / "docs" / "research_knowledge" / "index.md").read_text(encoding="utf-8")
    for direction, record in sorted(directions.items()):
        knowledge = ROOT / record.get("knowledge", "")
        if not knowledge.exists():
            failures.append(f"{direction}: missing knowledge doc {record.get('knowledge')}")
            continue
        knowledge_text = knowledge.read_text(encoding="utf-8")
        if direction not in index_text and knowledge.name not in index_text:
            failures.append(f"{direction}: not discoverable from docs/research_knowledge/index.md")
        for skill in record.get("skills", []):
            skill_path = SKILL_DIR / skill / "SKILL.md"
            if not skill_path.exists():
                failures.append(f"{direction}: missing skill {skill_path.relative_to(ROOT)}")
            if skill not in knowledge_text and skill not in index_text:
                failures.append(f"{direction}: skill {skill} not linked from index or {knowledge.relative_to(ROOT)}")
        if not record.get("memory_query"):
            failures.append(f"{direction}: missing memory_query")
        for runtime in record.get("runtime", []):
            if not (ROOT / runtime).exists():
                failures.append(f"{direction}: missing runtime path {runtime}")
        if not record.get("validation"):
            failures.append(f"{direction}: missing validation commands")

    if failures:
        print("check_research_task_router: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"check_research_task_router: OK directions={len(directions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
