# v19_reg Walk-Forward Review

Date: 2026-04-25
**CRITICAL UPDATE: 2026-04-25 — Fixed backtest window bug**

## Bug Description

`run_profile()` in `src/soxl_quant/app/train.py` called `backtest_profile()` without passing `test_seg`. This caused the default `TEST_SEG=("2020-01-01", "2026-03-31")` to be used for ALL walk-forward backtests, meaning the "walk-forward results" previously reported were actually 6-year backtests using walk-forward-trained actors, NOT isolated 1-year windows.

**Impact:** Previous walk-forward backtest numbers (+181.3%, +42.2%, -7.1%) are invalid. Worker `evaluate_split` metrics were correct all along.

**Fix:** `backtest_profile` now receives the correct `test_seg` derived from `test_df`.

## Corrected Results (Isolated Test Window Backtest)

| Window | v19_reg Return | v19_reg Sharpe | v19_reg MaxDD | v19 Return | v19 Sharpe | v19 MaxDD |
|--------|---------------|---------------|--------------|-----------|-----------|----------|
| wf_2020 | **+102.5%** | **1.92** | 20.8% | +105.9% | 2.04 | 20.3% |
| wf_2022 | **-30.7%** | **-0.87** | 39.4% | -30.7% | -0.87 | 39.4% |
| wf_2023_2024 (2024 only) | **-17.4%** | **-0.30** | 35.6% | — | — | — |
| Fixed-split proxy | +532.7% | 0.88 | 51.8% | +417.7% | 0.79 | 53.9% |

## Worker Eval vs Fixed Backtest

| Window | Worker Eval | Fixed Backtest | Gap |
|--------|------------|---------------|-----|
| wf_2022 (v19) | -25.2% | -30.7% | 5.5pp |
| wf_2022 (v19_reg) | -21.9% | -30.7% | 8.8pp |

**Worker eval is directionally correct and within ~9pp.** The previous 66pp gap was entirely due to the test_seg bug.

## Bear-Market Context

SOXL B&H 2022: **-86.6%**

Model losses of -23% to -31% represent **+55-63pp outperformance** vs buy-and-hold. This is successful bear-market defense, not failure.

## Comparison: v19 vs v19_reg

| Metric | v19 | v19_reg | Assessment |
|--------|-----|---------|------------|
| Fixed-split proxy | 417.7% / 0.79 | 532.7% / 0.88 | v19_reg better |
| wf_2020 | 105.9% / 2.04 / 20.3% | 102.5% / 1.92 / 20.8% | v19 slightly better |
| wf_2022 | -30.7% / -0.87 / 39.4% | -30.7% / -0.87 / 39.4% | **Identical** |
| wf_2024 | — | -17.4% / -0.30 / 35.6% | v19_reg poor |

**v19_reg does NOT improve walk-forward performance over v19.**
- wf_2020 return is slightly worse (102.5% vs 105.9%)
- wf_2022 is identical (-30.7%)
- wf_2024 is negative (-17.4%)
- Regularization (dropout 0.2, weight_decay 1e-5) provides no walk-forward benefit

## Decision

- `balanced_rule_residual_v19_reg`: **candidate (walk-forward blocked)** — regularization improves fixed-split proxy Sharpe but does not improve (and slightly degrades) walk-forward performance
- The v19 baseline remains equal or stronger in walk-forward

## Root Cause Analysis

1. **test_seg bug fixed** — All prior walk-forward backtest numbers were invalid. Worker eval was correct.
2. **v19_reg identical to v19 in 2022** — Suggests regularization did not learn a meaningfully different policy for bear markets.
3. **wf_2024 negative** — Models fail in 2024, suggesting regime-specific overfit or missing features.

## Next Actions

1. ✅ **Fix applied** — test_seg now passed correctly; all future walk-forward results are valid
2. **Do NOT promote v19_reg** — v19 baseline is equal or stronger in walk-forward
3. **Investigate v19_reg wf_2024 failure** — -17.4% in 2024 needs analysis
4. **Consider v19_reg_light** — Lighter regularization (dropout 0.1, weight_decay 1e-6) in progress
