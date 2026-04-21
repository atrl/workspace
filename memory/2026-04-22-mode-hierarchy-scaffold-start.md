---
date: '2026-04-22'
title: 2026-04-22 Mode Hierarchy Scaffold Start
kind: decision
profiles:
- balanced_rule_residual_mode_v1
- soxl
status: active
tags:
- active
- balanced_rule_residual_mode_v1
- decision
- execution-overlay
- soxl
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: 'The repo hit a clear optimization plateau on threshold, cooldown, and overlay
  retuning. The next step moved to a larger mechanism change: persistent high-level
  trading modes constraining the existing residual actor, rather than further sing'
---

# 2026-04-22 Mode Hierarchy Scaffold Start

## Context

The repo hit a clear optimization plateau on threshold, cooldown, and overlay
retuning. The next step moved to a larger mechanism change: persistent
high-level trading modes constraining the existing residual actor, rather than
further single-layer parameter drift.

## Evidence

- Plan:
  - `docs/superpowers/plans/2026-04-21-mode-hierarchy-v1.md`
- Workpad:
  - `codex_exec_plans/active/2026-04-10-soxl-agent-loop.md`

Implemented scaffold only in this loop:

- `ModeRouterConfig` added to config loading
- new `src/soxl_quant/policy/modes.py`
- mode-band support added to `positioning.py`
- `rebalance_threshold(..., mode=...)` hook added in `execution.py`
- temporary focused test file was used during the first scaffold pass, then
  removed after the user requested no new unit tests for this line of work

Validation:

- `PYTHONPATH=. qlib_venv/bin/python tests/test_position_cap.py`

The position-cap regression check passed after fixing one legacy no-mode
drawdown-cap regression. Future validation for this line should rely on model
backtests / harness outputs rather than unit tests.

## Decision

- Keep the scaffold.
- Do not claim strategy improvement yet; no training/backtest path uses the mode
  state yet.

## Next Action

Wire `ModeState` through environment, evaluation, backtest, and runtime, then
create the first candidate profile package `balanced_rule_residual_mode_v1`.
