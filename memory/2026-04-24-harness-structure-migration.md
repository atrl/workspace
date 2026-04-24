---
date: '2026-04-24'
title: 2026-04-24 harness structure migration
kind: ops
profiles:
- soxl
status: kept
tags:
- kept
- ops
- soxl
artifacts:
- evidence/index.json
workpad: codex_exec_plans/active/current.md
summary: Split the oversized active workpad into compact current state plus completed
  archives, separated research knowledge by mechanism category, introduced data and
  runtime indexes, and narrowed experiment packages back toward concise cards. The
  repo-local .codex skill rename was attempted but filesystem approval blocked writes
  to .codex/skills in this session.
---

# 2026-04-24 harness structure migration

## Context

The active workpad had grown past 4,900 lines, the experiment landscape mixed unrelated mechanisms, `experiments/` duplicated runtime and memory narratives, and data paths were split across root CSVs, `data/generalization`, `data/multi_asset`, and `qlib_soxl_binary`.

## What changed

- Added compact startup workpad: `codex_exec_plans/active/current.md`.
- Moved historical active workpads into `codex_exec_plans/completed/2026-04/`.
- Split mechanism-level conclusions into `docs/research_knowledge/`.
- Replaced `docs/experiment-landscape-review.md` with a compact landing page and kept the legacy narrative at `docs/research_knowledge/experiment-landscape-review-legacy.md`.
- Added data registry layout and generated `data/datasets.yaml`.
- Added `evidence/index.json` and later clarified runtime as execution state only.
- Added validation scripts for research knowledge, experiment cards, and data registry.
- Narrowed `experiments/candidate/` according to registry status and generated missing active card files.

## Validation

- `PYTHONPATH=. qlib_venv/bin/python scripts/check_research_knowledge.py`
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_experiment_cards.py`
- `PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py --check`

All three targeted checks passed before the full harness validation step.

## Blocker

The planned `.codex/skills/soxl-factor-*` rename to generic `leveraged-etf-*` skill paths could not be completed in this session because writes and renames under `.codex/skills/` returned `Operation not permitted`, and escalated approval timed out. Keep this as the next manual or approved migration step.

## Next Use

Start future repo orientation from `codex_exec_plans/active/current.md`, then `docs/research_knowledge/index.md`, then `memory/index.json` through `scripts/memory_search.py`.
