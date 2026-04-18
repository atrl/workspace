---
date: '2026-04-18'
title: RL Policy Module Migration
kind: failure
profiles:
- moe
status: dropped
tags:
- dropped
- failure
- moe
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: 'Legacy memory note normalized into the repo-wide schema. Date: 2026-04-18
  Context: package folders under `src/soxl_quant/` existed, but `src/soxl_quant/policy/*`
  was still mostly wrapper code pointing back into `src/soxl_quant/rl.py` that m'
---

# RL Policy Module Migration

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

- Date: 2026-04-18
- Context:
  - package folders under `src/soxl_quant/` existed, but `src/soxl_quant/policy/*` was still mostly wrapper code pointing back into `src/soxl_quant/rl.py`
  - that meant the apparent directory split did not reduce coupling or make model/config ownership clearer
- What changed:
  - added `src/soxl_quant/policy/environment.py` and moved the training simulator, state interpreter, action interpreter, tranche ledger, and episode sampling there
  - moved reward logic into `src/soxl_quant/policy/reward.py`
  - moved policy-network/model code into `src/soxl_quant/policy/models/moe.py` and `src/soxl_quant/policy/models/tide.py`
  - moved policy assembly and actor inference into `src/soxl_quant/policy/policy_builder.py`
  - moved fast split evaluation into `src/soxl_quant/policy/evaluation.py`
  - reduced `src/soxl_quant/rl.py` to a compatibility facade that re-exports the migrated symbols
- Validation:
  - `PYTHONPATH=. qlib_venv/bin/python -m py_compile ...` for migrated files: passed
  - import compatibility probe: `imports-ok True True True True`
  - `PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py`: `passed=32 failures=0`
- Important follow-up:
  - `RESULT_PNG` is still live and should not be deleted as dead code yet; it is used by `src/soxl_quant/backtest.py` and still reported by the legacy CLI
  - remaining cleanup should focus on import call sites (`cli.py`, `backtest.py`, `runtime.py`) and on consolidating artifact path ownership under `core/`

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
