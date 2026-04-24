---
date: 2026-04-19
title: RL Facade Detached From Scripts
kind: other
profiles: []
status: reference
tags:
- other
- reference
artifacts: []
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: Moved multi-asset source prep and staged training into dedicated app modules,
  updated repo scripts to import policy/app layers directly, and reduced internal
  dependence on the rl compatibility facade to zero.
---

# RL Facade Detached From Scripts

## Context

The previous step removed `rl.py` from internal code and tests, but several repo-owned scripts still imported it. That kept the compatibility facade in the active maintenance path.

## What Changed

- added `src/soxl_quant/app/sources.py`
- added `src/soxl_quant/app/selection.py`
- rewired `app/train.py`, `app/cli.py`, `app/workers.py`, and top-level `cli.py` to use the new app submodules
- updated repo scripts to import `policy.*` or `app.*` directly
- repaired `scripts/feature_analysis.py` to use current policy/environment names

## Evidence

- `rg -n "src\\.soxl_quant\\.rl|from \\.rl import" src scripts tests`: no matches
- `PYTHONPATH=. qlib_venv/bin/python -m py_compile ...`: passed
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py`: `passed=32 failures=0`

## Decision

Treat `rl.py` as legacy external compatibility only. Repo-owned scripts should be considered internal and must import `policy/` or `app/` directly.
