---
date: '2026-04-18'
title: 2026-04-18 Package Structure Refactor
kind: decision
profiles:
- moe
status: reference
tags:
- decision
- moe
- reference
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: 'The repository had already started splitting responsibilities conceptually,
  but the filesystem still looked flat. That made the intended architecture hard to
  see and easy to ignore. Added new package directories under `src/soxl_quant/`: `ap'
---

# 2026-04-18 Package Structure Refactor

## Context

The repository had already started splitting responsibilities conceptually, but
the filesystem still looked flat. That made the intended architecture hard to
see and easy to ignore.

## Evidence

- Added new package directories under `src/soxl_quant/`:
  - `app/`
  - `core/`
  - `policy/`
- Added compatibility modules in those directories so new code can import from
  structured locations immediately:
  - `app/cli.py`, `app/train.py`, `app/evaluate.py`, `app/promote.py`
  - `core/constants.py`, `core/artifacts.py`, `core/config.py`,
    `core/registry.py`, `core/dataset.py`
  - `policy/observation.py`, `policy/reward.py`, `policy/execution.py`,
    `policy/positioning.py`, `policy/policy_builder.py`, `policy/evaluation.py`
  - `policy/models/mlp.py`, `policy/models/moe.py`, `policy/models/tide.py`
- Switched package entrypoints to the new app layer:
  - `src/soxl_quant/__init__.py`
  - `soxl_qlib_strategy_v2.py`

## Decision

- Keep old flat modules as compatibility wrappers for now.
- New code should import from `app/`, `core/`, and `policy/` first.
- A later pass can move actual implementations behind these paths module by
  module once import churn is lower.
