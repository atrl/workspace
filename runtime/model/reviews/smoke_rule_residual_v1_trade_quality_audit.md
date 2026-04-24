# Trade Quality Audit: smoke_rule_residual_v1

## Summary

- profile: `smoke_rule_residual_v1`
- portfolio-level retained result: `test_proxy = 600.81% / Sharpe 0.896 / MaxDD 53.42%`
- `test_2022 = -18.39% / Sharpe -0.932 / MaxDD 25.31% / avg_pos 0.199`
- fixed-split promotion gate: `pass`

The actor remains usable at the portfolio level, but the trade-quality breakdown shows that short-horizon T behavior is not where the edge comes from.

## Kelly By Holding Period

- `short_<3d`: `9` trades | `Kelly 0.2052` | `Expectancy 2.31%`
- `mid_3_10d`: `65` trades | `Kelly 0.3143` | `Expectancy 3.64%`
- `long_10_30d`: `20` trades | `Kelly 0.1968` | `Expectancy 4.73%`
- `very_long_30d+`: `2` trades | `Kelly 0.0000` | `Expectancy -40.66%`

## Kelly By Position Change

- `micro_<5%`: `56` trades (`58.3%` of all trades) | `Kelly 0.0585` | `Expectancy 0.73%`
- `small_5_15%`: `19` trades | `Kelly 0.2013` | `Expectancy 2.89%`
- `large_15%+`: `21` trades | `Kelly 0.5491` | `Expectancy 8.34%`

## Friction And Interpretation

- total trade actions: `316`
- rough friction estimate: `31.60%`
- removing `micro_<5%` would save about `11.20%` of friction

## What The Model Learned

The current reward is dominated by step-level portfolio return, turnover/smoothness cost, drawdown/downside/risk-budget control, and inactivity/opportunity-cost shaping.
That means the actor is mainly optimized for aggregate path quality, not for per-trade round-trip expectancy.

## Conclusion

- `mid_3_10d` quality is `positive`: `Kelly 0.3143 / Expectancy 3.64%`
- `small_5_15%` quality is `positive`: `Kelly 0.2013 / Expectancy 2.89%`
- `micro_<5%` quality is `weak` and still frequency-heavy: `Kelly 0.0585 / Expectancy 0.73%`
- the residual overlay materially improved mid-horizon trade quality, but micro rebalancing is still too weak relative to its frequency and friction load

## Next Action Order

- raise the effective minimum rebalance threshold to suppress `micro_<5%` churn
- add or tighten `trade_lock` / `cooldown` for `3-10d` churn
- test fixed-actor low-micro replays before changing reward or architecture
