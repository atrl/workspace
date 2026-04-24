---
date: '2026-04-17'
title: '2026-04-17: `balanced_floor40_sym_bear30` retrain'
kind: review
profiles:
- balanced
- balanced_floor40_sym
- balanced_floor40_sym_bear30
- soxl
status: reference
tags:
- balanced
- balanced_floor40_sym
- balanced_floor40_sym_bear30
- checkpoint-selection
- eval-2022
- reference
- review
- soxl
- stress-regime
artifacts:
- artifacts/models/soxl_balanced_floor40_sym_bear30_summary.json
- runtime/model/reviews/floor40_sym_bear30_vs_canonical_balanced.md
- runtime/model/reviews/floor40_sym_bear30_vs_floor40_sym.md
workpad: memory/index.json
summary: Legacy memory note normalized into the repo-wide schema. `qlib_venv/bin/python
  soxl_qlib_strategy_v2.py --profiles balanced_floor40_sym_bear30 --skip-data-refresh
  --seed 20260411` Added `configs/soxl/balanced_floor40_sym_bear30.yaml`. Regis
---

# 2026-04-17: `balanced_floor40_sym_bear30` retrain

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Command

- `qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_floor40_sym_bear30 --skip-data-refresh --seed 20260411`

## What changed

- Added `configs/soxl/balanced_floor40_sym_bear30.yaml`.
- Registered the profile in `src/soxl_quant/cli.py`.
- This profile is a real retrain of `balanced_floor40_sym` with tighter stress-side de-risking:
  - intended change: `bear_min_cap = 0.30`
- Fixed a harness evidence gap in `src/soxl_quant/cli.py`:
  - model bundle and summary now persist the full dynamic `position_cap` payload, not only the base drawdown fields

## Result

- Full backtest log:
  - `533.9% / Sharpe 0.82 / MaxDD 62.2% / 786 trades`
- Summary:
  - `artifacts/models/soxl_balanced_floor40_sym_bear30_summary.json`

## Comparison vs `balanced_floor40_sym`

- `test_proxy`:
  - return `443.67% -> 442.96%`
  - Sharpe `0.7941 -> 0.7935`
  - maxDD `61.55% -> 61.05%`
- `test_2022`:
  - return `-36.99% -> -35.30%`
  - maxDD `45.35% -> 44.47%`
  - avg_pos `0.429 -> 0.394`
- `test_2023_2024`:
  - return `51.17% -> 44.96%`
  - maxDD `39.48% -> 41.74%`

## Comparison vs canonical `balanced`

- canonical `balanced` still wins on generality and promotion fitness:
  - `test_proxy Sharpe 0.858 > 0.793`
  - `test_proxy maxDD 50.18% < 61.05%`
  - `test_2022 maxDD 33.04% < 44.47%`
  - `test_2022 avg_pos 0.283 < 0.394`
- `balanced_floor40_sym_bear30` remains promotion-blocked.

## Main takeaway

- The earlier fixed-actor counterfactual was directionally correct:
  - stress-side floor tightening really is the live lever
- But the retrained branch shows that this lever is only incremental:
  - it slightly repairs `2022`
  - it does not change the branch into a general winner
- The next optimization target should be checkpoint selection / stress routing / actor style, not another tiny floor tweak.

## Evidence

- `runtime/model/reviews/floor40_sym_bear30_vs_floor40_sym.md`
- `runtime/model/reviews/floor40_sym_bear30_vs_canonical_balanced.md`

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
