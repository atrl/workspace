---
date: '2026-04-18'
title: 2026-04-18 stress-release delta feature trial
kind: failure
profiles:
- balanced
- balanced_stress_release_delta
status: dropped
tags:
- balanced
- balanced_stress_release_delta
- checkpoint-selection
- dropped
- eval-2022
- execution-overlay
- failure
- stress-regime
- transition-features
artifacts: []
workpad: memory/index.json
summary: 'Legacy memory note normalized into the repo-wide schema. Profile: `balanced_stress_release_delta`
  Command: `qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_stress_release_delta
  --skip-data-refresh --seed 20260411` Feature'
---

# 2026-04-18 stress-release delta feature trial

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## What ran

- Profile: `balanced_stress_release_delta`
- Command:
  - `qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_stress_release_delta --skip-data-refresh --seed 20260411`
- Feature change from canonical `balanced`:
  - `regime_risk_delta_5`
  - `vol_stress_score_delta_5`
  - `downside_ratio_10_delta_5`

## Result vs canonical `balanced`

- Validation score improved:
  - `61.42 -> 87.01`
- Fixed-split OOS degraded:
  - test proxy return `468.90% -> 459.62%`
  - test proxy Sharpe `0.858 -> 0.819`
  - test proxy maxDD `50.18% -> 57.44%`
  - `2022` return `-25.82% -> -37.38%`
  - `2022` maxDD `33.04% -> 45.51%`
  - `2022` avg_pos `0.283 -> 0.410`
- Promotion failed:
  - `test_2022_avg_pos`
  - `test_2022_max_dd`

## Good windows

- Small local rebound / restart improvements did appear:
  - `2022-07` excess `-52.98% -> -52.79%`, avg_pos `0.302 -> 0.293`
  - `2022-11` excess `-33.98% -> -32.31%`, avg_pos `0.433 -> 0.517`
  - `2023-01` excess `-33.32% -> -32.15%`, avg_pos `0.471 -> 0.481`

## Why it was dropped

- The narrow delta features are less harmful than the earlier broad transition bundle, but they still mostly express themselves as higher willingness to hold risk.
- The main failure pattern is unchanged:
  - validation improves sharply
  - rebound windows improve a little
  - full-window `2022` exposure and drawdown get materially worse
- Conclusion:
  - the missing transition signal is probably real
  - but simply adding transition-state features keeps steering the actor toward exposure drift instead of selective re-entry

## Next optimization order

1. Keep canonical `balanced` as the actor baseline.
2. Test fixed-actor execution/risk overlays before adding more features:
  - faster but state-conditional re-entry after stress release
  - tighter de-risking during high `regime_risk` / `vol_stress_score`
  - require no broad `2022` avg position increase
3. Review checkpoint selection / early stopping whenever validation rises sharply but fixed-split OOS degrades.

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
