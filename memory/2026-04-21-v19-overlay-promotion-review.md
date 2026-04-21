---
date: '2026-04-21'
title: 2026-04-21 v19 overlay promotion review
kind: review
profiles:
- balanced_rule_residual_v19
- soxl
status: kept
tags:
- balanced_rule_residual_v19
- eval-2022
- execution-overlay
- kept
- review
- soxl
- stress-regime
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: 'Base actor: `balanced_rule_residual_v19` Goal: compare the new dwell overlays
  against the existing SOXL overlay set on the same fixed-split promotion surface
  1. `dwell_hold_v2` 2. `dwell_hold_v1` 3. `regime_route_v1` 4. `stress_release_rout'
---

# 2026-04-21 v19 overlay promotion review

## Context

- Base actor: `balanced_rule_residual_v19`
- Goal: compare the new dwell overlays against the existing SOXL overlay set on
  the same fixed-split promotion surface

## Ranking

1. `dwell_hold_v2`
2. `dwell_hold_v1`
3. `regime_route_v1`
4. `stress_release_route_v1`
5. `base`

## Main Metrics

- `dwell_hold_v2`
  - proxy: `1061.2% / Sharpe 1.064 / MaxDD 44.42%`
  - `test_2022`: `-26.4% / MaxDD 34.73% / avg_pos 0.285`
  - `test_2023_2024`: `56.9% / MaxDD 37.20%`
- `regime_route_v1`
  - proxy: `808.0% / 0.994 / 49.66%`
  - `test_2022`: `-23.9% / 32.61% / avg_pos 0.272`
  - `test_2023_2024`: `61.8% / 37.70%`
- base
  - proxy: `774.1% / 0.968 / 51.79%`
  - `test_2022`: `-24.5% / 34.67% / avg_pos 0.279`
  - `test_2023_2024`: `46.1% / 38.74%`

## Readout

- `dwell_hold_v2` is the best SOXL-specific carry / persistence overlay in the
  current set
- `regime_route_v1` remains the better crisis-defense overlay
- this is no longer a single-axis leaderboard
- the frontier has split into:
  - persistence / lower churn / better proxy growth
  - or stronger `2022` defense

## Decision

- keep `dwell_hold_v2` and `regime_route_v1`
- stop treating `stress_release_route_v1` as additive here because it is
  identical to `regime_route_v1` on this surface

## Next Step

- run one narrow merge diagnostic:
  - start from `dwell_hold_v2`
  - add only stress-side asymmetry
  - see whether `2022` can improve without giving back the proxy gain
