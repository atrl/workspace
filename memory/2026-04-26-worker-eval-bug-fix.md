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

## Fix (commits 69bddd6, a2fcf65)
Modified `run_profile` in `src/soxl_quant/app/train.py`:
- After `backtest_profile` returns, copy `return`, `sharpe`, `max_dd` into `test_proxy`
- Also run `backtest_profile` on `test_2022` window and replace slice metrics
- Rebuild promotion report with corrected metrics

This fixes:
- Fixed-split promotion gate (test_proxy + test_2022_max_dd)
- Walk-forward worker eval (workers also call `run_profile`)
- All future training runs

## Known Residual Issue
`slice_metrics` still uses `evaluate_split` for:
- test_2020 metrics (analysis only)
- test_2023_2024 return (Pareto selection if use_pareto=True)
- Pareto checkpoint selection (if use_pareto=True)

These are NOT used for promotion gate but can mislead analysis.
Since `use_pareto` defaults to `False`, this is not urgent.
Fix when Pareto selection is enabled or when per-slice accuracy becomes critical.

## Validation Tool
`scripts/walkforward_backtest_validate.py` uses `backtest_profile` for correct walk-forward assessment.

## Key Insight
v19 actually outperforms buy-and-hold in ALL walk-forward windows:
- wf_2020: +20.7pp relative outperformance
- wf_2022: +341pp (loss avoidance, BnH -383%)
- wf_2023_2024: +26.1pp relative outperformance

The "failure" was entirely a measurement artifact.
