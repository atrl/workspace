---
date: '2026-04-17'
title: '2026-04-17: `balanced_regime_gate` failed as a mainline candidate'
kind: failure
profiles:
- balanced
- balanced_regime_gate
- soxl
status: dropped
tags:
- balanced
- balanced_regime_gate
- checkpoint-selection
- dropped
- eval-2022
- failure
- soxl
- stress-regime
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: 'Legacy memory note normalized into the repo-wide schema. New profile: `configs/soxl/balanced_regime_gate.yaml`
  Intent: keep canonical `balanced` as the style anchor add staged checkpoint scoring
  penalize validation stress-window average pos'
---

# 2026-04-17: `balanced_regime_gate` failed as a mainline candidate

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## What was tried

- New profile:
  - `configs/soxl/balanced_regime_gate.yaml`
- Intent:
  - keep canonical `balanced` as the style anchor
  - add staged checkpoint scoring
  - penalize validation stress-window average position
  - tighten bear-side cap only moderately

## Command

- `qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_regime_gate --skip-data-refresh --seed 20260411`

## Result vs canonical `balanced`

- Validation improved:
  - score `61.42 -> 79.66`
  - valid return `39.0% -> 53.5%`
- Out-of-sample degraded:
  - test proxy return `468.90% -> 373.48%`
  - test proxy Sharpe `0.858 -> 0.781`
  - test proxy maxDD `50.18% -> 55.56%`
- `2022` got materially worse:
  - return `-25.82% -> -39.22%`
  - maxDD `33.04% -> 43.41%`
  - avg_pos `0.283 -> 0.370`
- `2023-2024` improved:
  - return `43.52% -> 56.64%`
  - maxDD `33.87% -> 30.99%`

## Interpretation

- This branch is another exposure-drift failure:
  - better in uptrend / recovery
  - worse in stress
- The explicit `stress_avg_pos_penalty` in validation selection was not enough to align OOS behavior.
- The remaining problem is not “need slightly tighter bear cap”.
- The model still finds a way to keep too much trend exposure through `2022`.

## Next direction

- Keep staged checkpointing support.
- Do not keep this parameterization.
- Next branch should reduce trend-side encouragement first:
  - lower `activity_bonus`
  - lower `panic_buy_bonus`
  - increase selection penalties on `avg_pos` and `stress_avg_pos`
  - only then revisit cap thresholds if needed

## Decision

See the summary and status fields in front matter.

## Next Use

Use `runtime/index.json` before querying runtime knowledge before opening additional notes.
