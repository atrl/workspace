---
date: '2026-04-20'
title: 'v4 Results: Trade Compression Fails Again — Trades Increase to 327'
kind: experiment-result
profile: balanced_rule_residual_v4
baseline: balanced_rule_residual_v3
---

# v4 Results: Trade Compression Fails Again

## Summary

v4 training completed. Despite aggressive trade compression design (flat penalty 1.20, longer lock 8 days, reduced alpha 2.0), **trades increased from 314 to 327**. Strategy is highly profitable (730.8% return, 0.90 Sharpe) but trades far above 150-220 target. Promotion blocked.

## Results

| Metric | v2 | v3 | v4 |
|--------|-----|-----|-----|
| Test Return | 521.3% | 569.7% | **730.8%** |
| Test Sharpe | 0.86 | 0.86 | **0.90** |
| Test MaxDD | 50.4% | 53.6% | **55.7%** |
| Test Trades | 316 | 314 | **327** ❌ |
| Valid Best | 104.61 @40 | 101.07 @60 | **106.03 @80** |
| test_2020 | - | - | 116.7% / 2.45 / 18.4% |
| test_2022 | - | - | -25.0% / -0.91 / 33.2% |
| test_2023_2024 | - | - | 69.0% / 0.89 / 31.2% |

## Key Finding

The flat trade penalty of 1.20 per trade is **economically trivial** relative to the strategy's profitability. The agent makes 730.8% return — it can easily afford 327 × 1.20 = 392 points of penalty. The penalty is a cost of doing business, not a deterrent.

This mirrors a classic RL reward shaping problem: when the primary reward signal (returns) is much larger than the penalty, the agent optimizes around the penalty rather than being deterred by it.

## Why Trade Compression Keeps Failing

1. **Return signal dominates** — 730% return swamps any per-trade penalty < ~5-10
2. **Penalty is per-step, not per-episode** — agent optimizes locally at each step
3. **Backtest trade counting != environment step counting** — penalties shape policy but execution logic determines actual trades
4. **Hybrid overlay base drifts** — regime changes force position adjustments that count as trades

## What Worked

- Return improved dramatically (+161pp vs v3)
- Sharpe improved (0.86 → 0.90)
- All sub-periods strong (2020: +116%, 2023-24: +69%)
- 2022 drawdown acceptable (33.2% < 35% gate)

## What Didn't

- Trades went UP not down (314 → 327)
- The core problem remains unsolved

## v5 Hypothesis

To actually compress trades, the penalty must be **large enough to hurt**. Options:

1. **Massive flat penalty** (5.0-10.0) — make each trade genuinely expensive
2. **Cumulative penalty** — penalty increases with trade count per episode
3. **Hard trade cap** — limit trades per episode in environment
4. **Episode length reduction** — fewer steps = fewer trade opportunities

Recommended v5 approach: flat_trade_penalty = 5.0 (4x increase) + trade_lock_days = 12 (50% longer) + keep alpha_scale = 2.0 (proven to work for returns).

## Files

- Experiment: `experiments/candidate/balanced_rule_residual_v4/`
- Log: `log/balanced_rule_residual_v4.log`
- Model: `soxl_balanced_rule_residual_v4_actor.pt`
