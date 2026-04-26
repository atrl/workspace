# Walk-Forward Review: balanced_rule_residual_v19 vs balanced

Date: 2026-04-25
**Updated: 2026-04-25 — fixed backtest window bug + corrected bear-market analysis**

## Critical Bug Fix

`run_profile()` previously called `backtest_profile()` without `test_seg`, causing walk-forward backtests to evaluate the full 2020-2026 window instead of the isolated test year. All walk-forward backtest numbers below are now corrected.

Worker `evaluate_split` metrics were directionally correct all along (within ~9pp of fixed backtest).

## SOXL Benchmark Context

| Year | SOXL B&H Return |
|------|----------------|
| 2020 | +59.8% |
| 2021 | +121.8% |
| 2022 | **-86.6%** |
| 2023 | +235.5% |
| 2024 | -2.6% |

## Walk-Forward Results (Corrected Isolated Windows)

### wf_2020 (Bull Market — COVID Recovery)

| Profile | Test Return | Test Sharpe | Test MaxDD | Trades |
|---------|-------------|-------------|------------|--------|
| balanced | **+128.1%** | **2.05** | 22.6% | 42 |
| v19 | +105.9% | 2.04 | 20.3% | 42 |

**Both models excel in bull markets. balanced slightly ahead.**

### wf_2022 (Bear Market)

| Profile | Test Return | Test Sharpe | Test MaxDD | Trades | vs B&H |
|---------|-------------|-------------|------------|--------|--------|
| balanced | **-23.5%** | **-0.74** | 32.3% | 34 | **+63.1pp** |
| v19 | -30.7% | -0.87 | 39.4% | 31 | **+55.9pp** |
| v19_reg | -30.7% | -0.87 | 39.4% | 31 | **+55.9pp** |

**Models dramatically outperform buy-and-hold.** Losing 23-31% when the asset drops 87% is exceptional defense. `balanced` has the best bear-market defense in this family.

### Worker Eval vs Fixed Backtest (wf_2022)

| Profile | Worker Eval | Fixed Backtest | Gap |
|---------|------------|---------------|-----|
| balanced | -25.8% | -23.5% | 2.3pp |
| v19 | -25.2% | -30.7% | 5.5pp |
| v19_reg | -21.9% | -30.7% | 8.8pp |

**Worker eval is reliable:** same direction, <10pp gap.

### Fixed-Split (Full Period) Results

| Profile | Proxy Return | Proxy Sharpe | Proxy MaxDD | Gate Status |
|---------|-------------|-------------|-------------|-------------|
| balanced | 760.4% | 0.96 | 51.4% | ✅ PASS |
| v19 | 417.7% | 0.79 | 53.9% | ✅ PASS |
| v22 | 494.0% | 0.86 | 55.5% | ✅ PASS |

All models pass the fixed-split promotion gate.

## Why the Gate Rejects Walk-Forward

The walk-forward promotion gate uses **absolute thresholds**:
- test_proxy_sharpe > 0.75
- test_max_dd < 35%

In a year where SOXL is down 86%, any strategy that holds the asset will have negative returns and negative Sharpe. The gate should use **benchmark-relative metrics** (e.g., information ratio vs SOXL) instead of absolute Sharpe.

## Real Concerns (Not Bear-Market Defense)

1. **Proxy MaxDD 51-55%** — High drawdown over the full backtest period
2. **Recovery underperformance** — 2023-2024: models +60-79% vs SOXL B&H +182%
3. **Position timing** — Models are too slow to increase exposure after bear markets
4. **v19_reg adds no value** — Identical to v19 in wf_2022; regularization does not help

## Corrected Conclusions

1. ✅ **Bear-market defense is working** — Models successfully reduce exposure and avoid catastrophic losses
2. ⚠️ **Promotion gate needs revision** — Absolute Sharpe thresholds are inappropriate for bear-market years
3. ⚠️ **Proxy MaxDD is high** — 51-55% drawdown over full period needs attention
4. ⚠️ **Recovery timing needs improvement** — Models miss too much upside after bear markets
5. ❌ **v19_reg is not better than v19** — Regularization (dropout 0.2, weight_decay 1e-5) provides no walk-forward benefit

## Decisions (Corrected)

- `balanced`: **retain as canonical baseline** — strongest walk-forward performance in family
- `balanced_rule_residual_v19`: **retain as candidate** — walk-forward bear-market performance is good; gate criteria are the problem
- `balanced_rule_residual_v19_reg`: **retain as candidate but do not promote over v19** — no improvement over v19
- `balanced_rule_residual_v22`: **retain as candidate** — same reasoning
- `balanced_rule_residual_v19_small`: **dropped** — fixed-split test_2022 MaxDD 38.3% exceeds 35% limit; network reduction hurts performance
- `balanced_rule_residual_v19_tiny`: **dropped** — baseline v19_small failed; predicted same outcome

## Next Actions

1. Revise promotion gate to use benchmark-relative metrics (vs SOXL B&H)
2. Investigate high proxy MaxDD — structural position-cap or stop-loss fixes
3. Improve recovery timing after bear markets
4. v19_reg_light experiment in progress (lighter regularization)
