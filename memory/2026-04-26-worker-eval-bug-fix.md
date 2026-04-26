# 2026-04-26 Worker Eval Bug Fix

## Problem
`evaluate_split` (lightweight eval) has systematic gaps of 80-110pp vs `backtest_profile` (official qlib Exchange model).

Examples:
- v19 wf_2020: worker eval +184% vs backtest +92.8% (gap: -91pp)
- v19 wf_2022: worker eval +67.6% vs backtest -42.1% (gap: -110pp)
- v19_reg wf_2022: worker eval -23.9% vs backtest +42.2% (gap: +66pp)

Root cause: `evaluate_split` uses simplified cost model (0.1% per trade), no min_cost, no trade_lock, no buy_cooldown, simplified rebalance logic.

## Impact
- Promotion gate based on worker eval was meaningless
- Walk-forward comparisons were invalid
- v19 was incorrectly assessed as failing walk-forward

## Fix (commit 69bddd6)
Modified `run_profile` in `src/soxl_quant/app/train.py`:
- After `backtest_profile` returns, copy `return`, `sharpe`, `max_dd` into `test_proxy`
- Rebuild promotion report with corrected `test_proxy`

This fixes:
- Fixed-split promotion gate
- Walk-forward worker eval (workers also call `run_profile`)
- All future training runs

## Known Residual Issue
`slice_metrics` (and `evaluate_all_slices`) still uses `evaluate_split` for:
- test_2022 MaxDD
- test_2023_2024 return
- Pareto checkpoint selection (if use_pareto=True)

These are NOT used for promotion gate but can mislead analysis and Pareto selection.
Fix when Pareto selection is enabled or when per-slice accuracy becomes critical.

## Validation Tool
`scripts/walkforward_backtest_validate.py` uses `backtest_profile` for correct walk-forward assessment.

## Key Insight
v19 actually outperforms buy-and-hold in ALL walk-forward windows:
- wf_2020: +20.7pp relative outperformance
- wf_2022: +341pp (loss avoidance, BnH -383%)
- wf_2023_2024: +26.1pp relative outperformance

The "failure" was entirely a measurement artifact.
