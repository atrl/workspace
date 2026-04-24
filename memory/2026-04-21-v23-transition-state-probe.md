---
date: '2026-04-21'
title: 2026-04-21 v23 narrow transition-state probe
kind: failure
profiles:
- balanced_rule_residual_v19
- balanced_rule_residual_v23
- balanced_stress_release_delta
- balanced_transition_state
- smoke
status: dropped
tags:
- balanced_rule_residual_v19
- balanced_rule_residual_v23
- balanced_stress_release_delta
- balanced_transition_state
- checkpoint-selection
- dropped
- eval-2022
- execution-overlay
- failure
- smoke
- stress-regime
- transition-features
artifacts: []
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: Trained `balanced_rule_residual_v23` as a narrow 6-feature transition-state
  clone of v19 under `SOXL_USE_TRANSITION_V23=1`. Fixed-split gate still passed, but
  the model became broadly more defensive rather than selectively better at stress-release.
  `test_2022` improved slightly while `test_2020`, `test_2023_2024`, and proxy return
  dropped sharply. Best checkpoint was iter 0, indicating the new feature pack did
  not survive training pressure. Branch dropped and archived.
---

# 2026-04-21 v23 narrow transition-state probe

## Context

This was the first residual-family retry of transition-state features after the
repo had already dropped `balanced_transition_state` and
`balanced_stress_release_delta`. The explicit goal was to see whether a much
narrower pack could repair rebound / restart timing without reintroducing the
old `2022` exposure-drift failure.

## Run setup

```bash
SOXL_USE_TRANSITION_V23=1 PYTHONPATH=. qlib_venv/bin/python -m src.soxl_quant.cli \
    --profiles balanced_rule_residual_v23 --smoke --skip-data-refresh --seed 12345
```

- baseline: `balanced_rule_residual_v19`
- feature schema: `59 -> 65`
- signature: `99a98f3b024c -> f9d5cf71e7f7`
- new columns:
  - `vix_zscore_delta_5`
  - `downside_ratio_10_delta_5`
  - `days_above_stress_threshold_21`
  - `trend_slope_10`
  - `stress_then_relief_flag`
  - `trend_restart_flag`

## Key results vs v19

| slice | v19 | v23 | delta |
| --- | --- | --- | --- |
| valid return / sharpe / dd | 56.6 / 1.56 / 27.0 | 52.1 / 1.63 / 25.9 | lower return, slightly better risk |
| test_2020 return / dd | 109.4 / 18.1 | 91.6 / 16.2 | **return -17.8**, DD -1.9 |
| test_2022 return / dd / pos | -22.5 / 31.1 / 0.261 | -21.9 / 30.2 / 0.250 | slight defensive improvement |
| test_2023_2024 return / dd | 43.6 / 37.3 | 34.3 / 29.4 | **return -9.3**, DD -7.9 |
| test_proxy return / sharpe / dd | 735.1 / 0.95 / 54.8 | 493.2 / 0.89 / 51.2 | **return -241.9**, Sharpe -0.06, DD -3.6 |

Fixed-split promotion gates still passed:

- `test_2022_max_dd`: `30.23 < 35`
- `test_2022_avg_pos`: `0.250 < 0.35`
- `test_proxy_sharpe`: `0.887 > 0.75`

But the branch is not promotion-worthy because it gives up too much upside for
too little additional control.

## Selection signal

- best checkpoint: `iter 0`, score `81.59`
- iter 80 score: `75.56`

This is the clearest sign that the new transition pack does not improve the
trained policy. The architecture performs best before gradient updates move it
toward a more defensive style.

## Interpretation

- The transition-state signal is still real, but in this implementation it does
  **not** create selective re-entry.
- Instead, it behaves like another pathway to justify lower exposure.
- That is better than the older failure mode (`2022` exposure drift up), but it
  is still a failure for mainline promotion because the opportunity cost is too
  large.

## Decision

- Drop `balanced_rule_residual_v23`.
- Archive the experiment package.
- Keep summary + review evidence only.

## Next action

Return to the repo-wide optimization order:

1. fixed-actor execution / risk overlay before more transition-feature retrains
2. if retraining resumes, require a mechanism that can prove selective
   stress-release re-entry rather than broad de-risking
