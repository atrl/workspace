# Balanced Trade Quality Audit

Date: 2026-04-19
Profile: `balanced`
Primary artifacts:
- `artifacts/models/soxl_balanced_summary.json`
- `runtime/diagnosis/phase1_phase2_combined.json`
- `runtime/diagnosis/soxl_trade_attribution.md`
- `runtime/diagnosis/period_slice_comparison.md`
- `src/soxl_quant/policy/reward.py`

## Summary

The retained canonical `balanced` actor is still the best promotable model in the repo at the portfolio level:

- `test_proxy = 468.90% / Sharpe 0.858 / MaxDD 50.18%`
- fixed-split promotion gate: `pass`
- `test_2022 = -25.82% / Sharpe -0.877 / MaxDD 33.04% / avg_pos 0.283`

But the trade-quality audit shows that the model does **not** earn its result from consistently positive short-horizon T trades.

The main issue is not "the model learned nothing". The issue is narrower:

- it learned a usable **portfolio-path control policy**
- but it did **not** learn a uniformly positive **per-trade expectancy policy**
- weak and negative-expectancy buckets are concentrated in:
  - `mid_3_10d`
  - `small_5_15%`
  - a large mass of `micro_<5%` rebalances with near-zero Kelly

## Where Kelly Is Weak Or Negative

### By holding period

- `short_<3d`:
  - `85` trades
  - `Kelly 0.1573`
  - `Expectancy 0.93%`
  - weakly positive, but not strong enough to justify heavy turnover by itself
- `mid_3_10d`:
  - `57` trades
  - `Kelly -0.3287`
  - `Expectancy -3.02%`
  - the clearest negative-expectancy bucket in the strategy
- `long_10_30d`:
  - `26` trades
  - `Kelly 0.2968`
  - `Expectancy 4.87%`
  - materially better than the mid bucket
- `very_long_30d+`:
  - only `3` trades
  - all profitable
  - too few to treat as stable on their own, but directionally consistent with "large trend holds carry the edge"

### By position-change size

- `micro_<5%`:
  - `109` trades
  - `63.7%` of all trades
  - `Kelly 0.0437`
  - `Expectancy 0.38%`
  - near-zero edge after costs
- `small_5_15%`:
  - `57` trades
  - `Kelly -0.0624`
  - `Expectancy -0.52%`
  - negative edge
- `large_15%+`:
  - `5` trades
  - `Kelly 1.0`
  - `Expectancy 61.33%`
  - too few to overgeneralize, but the clear signal is that larger, rarer actions dominate quality

## Friction And Execution Interpretation

The trade count is too high for the quality of the small and medium buckets:

- total trade actions: `578`
- rough friction estimate at `0.1%` per side: `57.8%`
- removing `micro_<5%` trades alone would save about `21.8%` of friction

This is the central mismatch:

- `micro_<5%` trades are the majority of activity
- but they contribute only a minority of PnL
- their Kelly is almost flat

That means the actor is spending too much of its action budget on low-information rebalancing.

## What Reward The Model Actually Learned

The current reward in `src/soxl_quant/policy/reward.py` is dominated by:

- step-level portfolio return
- turnover and smoothness penalties
- drawdown / downside / risk-budget control
- inactivity and opportunity-cost pressure
- trend-alignment, trapped-loser, and stale-position penalties

This reward does **not** directly optimize:

- round-trip expectancy
- bucket-level Kelly
- cost-adjusted quality of specific T trades

So when `mid_3_10d` or `small_5_15%` buckets are negative, the right interpretation is:

- the actor learned when to adjust the **portfolio path**
- not necessarily when an individual short-horizon rebalance has positive standalone expectancy

In other words:

- the canonical model is good enough at aggregate path control to pass the gate
- but its T-style rebalances are partly a side-effect of the path objective, not a clean alpha source

## Why This Still Passes Promotion

At the portfolio level, the strategy still works because the strong behaviors dominate:

- 2022 defense remains acceptable:
  - `test_2022 maxDD 33.04%`
  - `test_2022 avg_pos 0.283`
- full-window proxy quality remains promotable:
  - `test_proxy Sharpe 0.858`

So the model is not broken. It is just inefficient.

The likely earnings stack is:

1. a handful of larger and longer holds capture the meaningful edge
2. broad de-risking logic reduces catastrophic exposure in stress
3. noisy short-horizon rebalances give back part of that edge through friction

## Period-Level Cross-Check

The period-slice diagnosis is consistent with the Kelly audit:

- `2022` excess is positive versus buy-and-hold because the model does de-risk
- `2023-2025` contains repeated underperformance in uptrending quarters
- the underperformance pattern is consistent with:
  - too little persistent core exposure in trends
  - too much micro-adjustment around the trend

This supports the same mechanism story:

- the model is stronger at avoiding full damage than at harvesting clean trend convexity
- weak T quality and underexposed trend capture are two sides of the same execution style

## Decision

Keep:

- the canonical `balanced` actor as the current retained baseline
- the conclusion that the actor has real path-level value
- the conclusion that trade compression is still a live optimization target

Do not conclude:

- that reward must be fully redesigned immediately
- that the actor needs a new architecture first

## Next Action Order

Run these as fixed-actor counterfactuals before any retrain:

1. raise the effective minimum rebalance threshold to suppress `micro_<5%` actions
2. add or tighten `trade_lock` / `cooldown` behavior for `3-10d` churn
3. test a no-micro or low-micro replay on the retained actor
4. only if those replays help directionally but not enough, add training-side reward terms for:
   - micro-rebalance penalty
   - short-hold flip penalty
   - round-trip or marginal-action reward

## Bottom Line

The negative Kelly buckets do not mean the canonical model failed to learn.

They mean it learned the wrong thing for T:

- good enough **aggregate path reward**
- weak **per-trade expectancy discipline**

That makes execution compression the next cheaper and more defensible lever than architecture changes or broad reward rewrites.
