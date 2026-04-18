---
date: '2026-04-17'
title: '2026-04-17: `balanced_regime_gate_v2` also failed'
kind: failure
profiles:
- balanced
- balanced_regime_gate
- balanced_regime_gate_v2
- soxl
status: dropped
tags:
- balanced
- balanced_regime_gate
- balanced_regime_gate_v2
- checkpoint-selection
- dropped
- eval-2022
- execution-overlay
- failure
- soxl
- stress-regime
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: 'Legacy memory note normalized into the repo-wide schema. New profile: `configs/soxl/balanced_regime_gate_v2.yaml`
  Intent: keep the moderate cap routing from `balanced_regime_gate` reduce trend-side
  encouragement directly: lower `activity_bo'
---

# 2026-04-17: `balanced_regime_gate_v2` also failed

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## What was tried

- New profile:
  - `configs/soxl/balanced_regime_gate_v2.yaml`
- Intent:
  - keep the moderate cap routing from `balanced_regime_gate`
  - reduce trend-side encouragement directly:
    - lower `activity_bonus`
    - lower `panic_buy_bonus`
    - weaken opportunity-cost penalties
  - increase validation penalties on `avg_pos` and `stress_avg_pos`

## Command

- `qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_regime_gate_v2 --skip-data-refresh --seed 20260411`

## Result vs canonical `balanced`

- Validation improved modestly:
  - score `61.42 -> 70.86`
- Out-of-sample degraded materially:
  - test proxy return `468.90% -> 360.82%`
  - test proxy maxDD `50.18% -> 62.89%`
- `2022` got worse again:
  - return `-25.82% -> -43.63%`
  - maxDD `33.04% -> 49.52%`
  - avg_pos `0.283 -> 0.368`

## Result vs `balanced_regime_gate`

- This did not rescue the first branch:
  - validation score `79.66 -> 70.86`
  - test proxy return `373.48% -> 360.82%`
  - test proxy maxDD `55.56% -> 62.89%`
  - `2022` return `-39.22% -> -43.63%`
  - `2022` maxDD `43.41% -> 49.52%`

## Interpretation

- The problem is not just "trend bonuses too high".
- Reward-weight tuning plus validation exposure penalties still does not transfer to fixed-split `2022`.
- Selection pressure on `avg_pos` and `stress_avg_pos` is not a reliable proxy for the true failure mode.

## Next direction

- Keep staged checkpointing and review tooling.
- Drop this parameterization.
- Do not spend the next loop on more small reward retunes of the same actor style.
- Move to structural levers:
  - regime-persistence / stress-state features
  - or execution-layer overlays / routing logic

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
