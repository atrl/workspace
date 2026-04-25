# v19_reg Walk-Forward Review

Date: 2026-04-25

## Critical Correction

**Previous analysis used worker internal evaluation metrics, which differ dramatically from actual backtest results.** This review uses the main script's backtest results (top-level worker JSON metrics).

## Results (Main Backtest)

| Window | v19_reg Return | v19_reg Sharpe | v19_reg MaxDD | v19 Return | v19 MaxDD |
|--------|---------------|---------------|--------------|-----------|-----------|
| wf_2020 | +181.3% | 0.59 | 76.7% | +184.0% | 76.7% |
| wf_2022 | **+42.2%** | 0.34 | 61.6% | **+67.6%** | 70.3% |
| wf_2023_2024 | -7.1% | 0.05 | 43.4% | — | — |
| Fixed-split | +532.7% | 0.88 | 51.8% | +417.7% | 53.9% |

## Key Finding: Models Make Money in 2022 Bear Market

**Both v19 and v19_reg show POSITIVE returns in 2022** according to the main backtest:
- v19: +67.6% (SOXL B&H: -86.6%)
- v19_reg: +42.2% (SOXL B&H: -86.6%)

This is exceptional bear-market performance. The models successfully navigate the 2022 crash.

## Why Promotion Is Blocked

The main script reports `promotion: blocked`. The block is NOT due to negative returns, but due to:

1. **Very high MaxDD** — wf_2020 MaxDD 76.7%, wf_2022 MaxDD 61.6%
2. **wf_2023_2024 negative return** — -7.1% with 43.4% MaxDD

## Worker Internal Evaluation vs Main Backtest

There's a massive discrepancy between worker internal metrics and main backtest:

| Window | Worker Eval | Main Backtest | Gap |
|--------|------------|--------------|-----|
| wf_2022 (v19) | -25.2% | +67.6% | **+92.8pp** |
| wf_2022 (v19_reg) | -23.9% | +42.2% | **+66.1pp** |

**The worker's internal evaluation is NOT a reliable predictor of backtest performance.** The evaluation method used during training differs significantly from the actual trading simulation.

## Comparison: v19 vs v19_reg

| Metric | v19 | v19_reg | Assessment |
|--------|-----|---------|------------|
| Fixed-split proxy | 417.7% / 0.79 | 532.7% / 0.88 | v19_reg better |
| wf_2020 | 184.0% / 76.7% | 181.3% / 76.7% | Similar |
| wf_2022 | 67.6% / 70.3% | 42.2% / 61.6% | v19 better return, worse MaxDD |
| wf_2023_2024 | — | -7.1% / 43.4% | v19_reg poor |

**v19_reg does NOT improve walk-forward performance over v19.**
- wf_2022 return is worse (42.2% vs 67.6%)
- wf_2023_2024 is negative (-7.1%)
- MaxDD improvement (61.6% vs 70.3%) is offset by worse returns

## Decision

- `balanced_rule_residual_v19_reg`: **candidate (walk-forward blocked)** — regularization improves fixed-split proxy Sharpe but hurts walk-forward performance
- The v19 baseline remains stronger in walk-forward

## Root Cause Analysis

1. **Worker eval ≠ Backtest** — The training evaluation systematically underestimates performance. This has been true for all models.
2. **High MaxDD is the real problem** — 61-77% drawdowns are unacceptable regardless of returns.
3. **wf_2023_2024 failure** — Models trained through 2022 fail in 2023-2024, suggesting 2023 has novel patterns.

## Next Actions

1. **Fix worker evaluation** — Align training evaluation with actual backtest methodology
2. **Address MaxDD** — 61-77% drawdowns need structural fixes (position caps, stop losses)
3. **Understand 2023 blind spot** — Analyze what regime shift in 2023 causes policy failure
4. **Do NOT promote v19_reg** — v19 baseline is stronger in walk-forward
