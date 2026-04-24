---
date: 2026-04-19
title: Internal RL Facade Shrink
kind: other
profiles: []
status: reference
tags:
- other
- reference
artifacts: []
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: Split artifact persistence and walk-forward workers out of app/train.py and
  updated repo-internal tests/imports so internal code no longer depends on the rl
  compatibility facade.
---

# Internal RL Facade Shrink

## Context

After the previous consolidation, `rl.py` was already a facade, but the next useful proof point was removing remaining repo-internal reliance on it and cutting `app/train.py` down further.

## What Changed

- added `src/soxl_quant/app/artifacts.py`
- added `src/soxl_quant/app/workers.py`
- reduced `src/soxl_quant/app/train.py` from the training plus persistence plus worker mix to primarily training and profile-run logic
- updated tests to import `policy.*` modules directly

## Evidence

- `PYTHONPATH=. qlib_venv/bin/python -m py_compile ...`: passed
- `rg -n "from src\\.soxl_quant\\.rl import|from \\.rl import" src tests`: no matches
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py`: `passed=32 failures=0`

## Decision

`rl.py` should now be treated as external compatibility only. Internal code should import from `policy/` and `app/` directly.
