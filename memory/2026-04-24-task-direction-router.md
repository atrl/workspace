---
date: 2026-04-24
title: Task direction router for research knowledge and skills
kind: ops
profiles: []
status: kept
tags:
- kept
- ops
artifacts:
- runtime/factor/analysis/
workpad: codex_exec_plans/active/current.md
summary: Added a task-direction router so future sessions classify work as factor,
  data, model_selection, network, reward, execution, or multi_asset before querying
  memory or opening runtime artifacts. Research knowledge now links directly to relevant
  skills and route-specific memory queries.
---

# Task direction router for research knowledge and skills

## Context

The previous cleanup still left an important ambiguity: `memory/` looked like a
parallel task surface, while `docs/research_knowledge/` and `.codex/skills/`
were not directly connected. That made it unclear whether the next iteration was
factor work, model-selection work, data refresh, execution testing, or another
category.

## Decision

Use task direction as the first decision:

- `factor`
- `data`
- `model_selection`
- `network`
- `reward`
- `execution`
- `multi_asset`

`memory/` is not a direction. It is queried only after the route is selected.

## Implementation

- Added `docs/research_knowledge/task_router.json` mapping each direction to:
  - knowledge file
  - skills
  - memory query terms
  - runtime paths
  - validation commands
- Rewrote `docs/research_knowledge/index.md` as the first-stop task router.
- Added `## Use With Skills` to each mechanism knowledge file.
- Added `scripts/check_research_task_router.py` and wired it into `scripts/check_harness.py`.
- Moved generated factor research CSV/YAML outputs from `docs/factor_research/` into `runtime/factor/analysis/`.
- Updated `AGENTS.md` so every task starts by declaring a primary direction.

## Validation

- `PYTHONPATH=. qlib_venv/bin/python scripts/check_research_knowledge.py`: OK.
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_research_task_router.py`: OK.

## Next Use

For future tasks, choose the direction first, then follow the route. Do not ask
"what memory should I read?" before choosing factor/model/data/execution/etc.
