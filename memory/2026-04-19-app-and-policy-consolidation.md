---
date: 2026-04-19
title: App And Policy Consolidation
kind: other
profiles: []
status: reference
tags:
- other
- reference
artifacts:
- runtime/README.md
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: Consolidated real train/evaluate/promote ownership under app/, moved execution/observation/positioning
  implementations under policy/, deleted top-level rl_* helper modules, and tightened
  runtime/memory/experiments index guidance.
---

# App And Policy Consolidation

## Context

`src/soxl_quant/policy/` and `src/soxl_quant/app/` existed, but several files were still wrappers pointing back to top-level modules. The tree looked cleaner than it really was and kept `cli.py` plus `rl_*` as hidden coupling points.

## What Changed

- moved execution, observation, and positioning implementations into the real `policy/` modules
- deleted top-level `rl_execution.py`, `rl_observation.py`, and `rl_positioning.py`
- moved train/evaluate/promote orchestration into real `app/` modules and reduced top-level `cli.py` to a compatibility facade
- updated `backtest.py` and `runtime.py` to import policy modules directly
- repaired `memory/README.md`, added `runtime/README.md`, and tightened `experiments/README.md` so readers start from indexes and summaries instead of browsing duplicate-looking files

## Evidence

- `PYTHONPATH=. qlib_venv/bin/python -m py_compile ...`: passed
- compatibility import probe: `imports-ok True True True True True True True`
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py`: `passed=32 failures=0`

## Decision

Keep `rl.py` and `cli.py` only as backward-compatible facades. New behavior should land in `policy/` or `app/` directly.

## Next Use

If `runtime/` grows another large family of artifacts, pair it with one compact summary markdown or index file instead of relying on raw logs alone.
