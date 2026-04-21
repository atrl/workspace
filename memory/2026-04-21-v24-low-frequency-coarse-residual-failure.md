---
date: '2026-04-21'
title: 2026-04-21 v24 low-frequency coarse residual failure
kind: failure
profiles:
- balanced_rule_residual_v19
- balanced_rule_residual_v24
status: dropped
tags:
- balanced_rule_residual_v19
- balanced_rule_residual_v24
- checkpoint-selection
- dropped
- eval-2022
- execution-overlay
- failure
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: 'Experiment: `balanced_rule_residual_v24` Baseline: `balanced_rule_residual_v19`
  Mechanism: `decision_frequency: 2` hybrid residual deltas compressed to `[-0.08,
  0.0, 0.08]` Intent: test whether medium-strength temporal abstraction can reduc'
---

# 2026-04-21 v24 low-frequency coarse residual failure

## Context

- Experiment: `balanced_rule_residual_v24`
- Baseline: `balanced_rule_residual_v19`
- Mechanism:
  - `decision_frequency: 2`
  - hybrid residual deltas compressed to `[-0.08, 0.0, 0.08]`
- Intent:
  - test whether medium-strength temporal abstraction can reduce drawdown and
    turnover without reopening feature-pack or overlay churn

## Result

- valid:
  - `61.2% / Sharpe 1.67 / MaxDD 27.3% / avg_pos 0.601`
- full test:
  - `439.9% / Sharpe 0.775 / MaxDD 58.35% / trades 176`
- slices:
  - `test_2022 = -22.24% / Sharpe -0.86 / MaxDD 30.67% / avg_pos 0.257`
  - `test_2023_2024 = 38.23% / Sharpe 0.66 / MaxDD 31.58%`
- proxy:
  - `758.6% / Sharpe 0.961 / MaxDD 54.58%`

## Comparison To v19

- small intended win:
  - `test_2022 max_dd 31.07 -> 30.67`
  - `test_2022 avg_pos 0.261 -> 0.257`
- main failure:
  - turnover did not compress
  - review trade markers expanded from roughly `35` on `v19` to `176`
- retained upside weakened:
  - `test_2023_2024 return 43.60 -> 38.23`
- proxy quality stayed almost unchanged:
  - `sharpe 0.946 -> 0.961`
  - `max_dd 54.83 -> 54.58`

## Harness Note

- first training run completed checkpoint selection but crashed while writing
  the summary sidecar because `persist_summary_bundle()` called
  `feature_schema_payload()` without passing `profile`
- fixed in `src/soxl_quant/app/artifacts.py`
- regression added in `tests/test_summary_bundle.py`
- actor artifact from the first run was valid and used to recover the summary
  and backtest without retraining

## Decision

- drop `v24`
- keep as the closest failure case for future temporal-abstraction work

## Rule For Next Time

- do not retry `decision_frequency` or coarse-delta compression by itself
- if temporal abstraction returns, it needs one of:
  - explicit dwell / action persistence constraints
  - hierarchical actor over slower rule states
  - execution-side accumulation / throttle logic
