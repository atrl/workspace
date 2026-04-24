---
date: '2026-04-18'
title: '2026-04-18: first transition-feature batch failed as mainline'
kind: failure
profiles:
- balanced
- balanced_transition_state
status: dropped
tags:
- balanced
- balanced_transition_state
- dropped
- eval-2022
- failure
- stress-regime
- transition-features
artifacts: []
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: 'Legacy memory note normalized into the repo-wide schema. New temporary branch:
  `balanced_transition_state` Intent: keep canonical `balanced` reward, cap, and selection
  rules fixed add transition / persistence features only test whether rebo'
---

# 2026-04-18: first transition-feature batch failed as mainline

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## What was tried

- New temporary branch:
  - `balanced_transition_state`
- Intent:
  - keep canonical `balanced` reward, cap, and selection rules fixed
  - add transition / persistence features only
  - test whether rebound / restart recognition improves without losing `2022` defense

## Command

- `qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_transition_state --skip-data-refresh --seed 20260411`

## Result vs canonical `balanced`

- Validation improved strongly:
  - score `61.42 -> 83.04`
- Full proxy quality degraded:
  - return `468.90% -> 472.85%`
  - Sharpe `0.858 -> 0.819`
  - maxDD `50.18% -> 58.97%`
- `2022` got materially worse:
  - return `-25.82% -> -40.96%`
  - maxDD `33.04% -> 46.45%`
  - avg_pos `0.283 -> 0.397`

## Targeted window readout

- Small local improvements appeared in a few restart windows:
  - `2022-07` excess `-29.78% -> -29.47%`
  - `2022-11` excess `-31.69% -> -28.28%`
  - `2023-01` excess `-23.06% -> -21.32%`
- But these came with broader exposure expansion:
  - `2022-11` avg_pos `0.433 -> 0.518`
  - `2023-01` avg_pos `0.471 -> 0.539`

## Interpretation

- Transition awareness is a real missing dimension.
- But this first feature batch encoded it too bluntly:
  - it raised risk appetite broadly instead of only improving transition timing
- This is a useful negative result:
  - "add transition features" was directionally right
  - "add a wide batch directly into the base schema" was the wrong implementation

## Keep / Drop

- Drop the transition feature batch from the shared default code path.
- Keep the evidence and the new daily replay export tool:
  - `scripts/export_profile_daily_backtest.py`

## Next direction

- Revisit transition features in smaller gated batches or profile-local ablations.
- Prefer one narrow family at a time:
  - stress-release deltas
  - or persistence counters
  - or restart flags
- Do not widen the global base schema again until one family proves it can improve `2022` without broad exposure drift.

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
