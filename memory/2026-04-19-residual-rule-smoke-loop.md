---
date: 2026-04-19
title: 2026-04-19 residual-over-rule smoke loop
kind: decision
profiles:
- balanced
- smoke
- smoke_rule_residual_v1
status: kept
tags:
- balanced
- decision
- eval-2022
- execution-overlay
- kept
- rule-based
- smoke
- smoke_rule_residual_v1
artifacts: []
workpad: memory/index.json
summary: Implemented and smoke-tested a residual-over-rule path. The actor now learns
  a bounded delta around the rule-based base position, with rule-deviation and micro-churn
  penalties. The first smoke candidate materially improved the previously negative
  `3-10d` trade bucket to positive Kelly/expectancy and reduced total trade actions
  from the prior canonical diagnosis baseline, but `micro_<5%` rebalances remain weak
  and too frequent.
---

# 2026-04-19 residual-over-rule smoke loop

## Context

The user asked for a direct trial of the "rule-based anchor + RL residual T overlay" idea rather than another abstract proposal.

## Evidence

- Added unified hybrid overlay mapping across:
  - training
  - validation
  - qlib backtest
  - runtime replay
- Added reward-side controls for:
  - rule deviation
  - micro rebalance discipline
  - short-hold turnover discipline
- Added candidate experiment package:
  - `experiments/candidate/smoke_rule_residual_v1/`
- Added reusable tooling:
  - `scripts/trade_quality_audit.py`
  - generalized `scripts/diagnosis_phase1_phase2.py`
  - upgraded `scripts/export_profile_daily_backtest.py`

## Results

- Smoke summary:
  - `valid = 70.23% / Sharpe 1.784 / MaxDD 27.33%`
  - `test_2022 = -18.39% / Sharpe -0.932 / MaxDD 25.31% / avg_pos 0.199`
  - `test_2023_2024 = 55.23% / Sharpe 0.835 / MaxDD 29.06% / avg_pos 0.429`
  - `test_proxy = 600.81% / Sharpe 0.896 / MaxDD 53.42% / avg_pos 0.522`
- Trade-quality diagnosis from the profile's own exported `detailed_trades.csv`:
  - `mid_3_10d = Kelly 0.3143 / Expectancy 3.64%`
  - `small_5_15% = Kelly 0.2013 / Expectancy 2.89%`
  - `micro_<5% = 56 trades / Kelly 0.0585 / Expectancy 0.73%`
  - total trade actions `316`
  - rough friction estimate `31.6%`

## Decision

- Keep the residual-over-rule direction.
- Keep the profile-specific export -> diagnosis -> audit chain; it is now strong enough to reuse.
- Do not claim success on micro churn yet; that remains the next bottleneck.

## Next action

- Escalate the same structure to a balanced-family quick profile.
- Tighten rebalance thresholds and trade lock before broadening delta levels.
