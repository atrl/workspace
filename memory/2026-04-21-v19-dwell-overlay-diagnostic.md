---
date: '2026-04-21'
title: 2026-04-21 v19 dwell overlay diagnostic
kind: decision
profiles:
- balanced_rule_residual_v19
- soxl
status: reference
tags:
- balanced_rule_residual_v19
- decision
- execution-overlay
- reference
- soxl
- stress-regime
artifacts: []
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: 'Base actor: `balanced_rule_residual_v19` Motivation: `v24` showed that lower-frequency
  coarse retraining did not reduce realized churn, so the next cheap question was
  whether the missing ingredient was execution-side persistence. `dwell_hol'
---

# 2026-04-21 v19 dwell overlay diagnostic

## Context

- Base actor: `balanced_rule_residual_v19`
- Motivation: `v24` showed that lower-frequency coarse retraining did not reduce
  realized churn, so the next cheap question was whether the missing ingredient
  was execution-side persistence.

## Variants Tested

- `dwell_hold_v1`
- `dwell_hold_v2`
- `stress_asym_hold_v1`

All run through:

- `scripts/execution_overlay_generalization.py`

## SOXL Result

- base:
  - `412.7% / Sharpe 0.837 / MaxDD 48.79% / 313 trades`
- `dwell_hold_v1`:
  - `390.3% / 0.813 / 47.35% / 176 trades`
- `dwell_hold_v2`:
  - `418.4% / 0.822 / 48.30% / 135 trades`
- `stress_asym_hold_v1`:
  - `435.5% / 0.866 / 46.72% / 372 trades`

## Readout

- `dwell_hold_v2` is the key cheap win:
  - large trade compression
  - SOXL return slightly above base
  - drawdown roughly unchanged
- `dwell_hold_v1` is safer as a generalized compression profile, but weaker on
  SOXL upside
- `stress_asym_hold_v1` improves SOXL return and DD, but increases churn and is
  therefore not a real persistence solution

## Conclusion

- temporal compression on this branch is more naturally an execution problem
  than a retraining problem
- do not open another `decision_frequency` retrain yet
- first push fixed-actor persistence overlays farther, especially around
  `dwell_hold_v2`
