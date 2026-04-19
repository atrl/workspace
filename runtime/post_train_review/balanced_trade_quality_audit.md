# Trade Quality Audit: balanced

## Summary

- profile: `balanced`
- portfolio-level retained result: `test_proxy = 468.90% / Sharpe 0.858 / MaxDD 50.18%`
- `test_2022 = -25.82% / Sharpe -0.877 / MaxDD 33.04% / avg_pos 0.283`
- fixed-split promotion gate: `pass`

The actor remains usable at the portfolio level, but the trade-quality breakdown shows that short-horizon T behavior is not where the edge comes from.

## Kelly By Holding Period

- `short_<3d`: `85` trades | `Kelly 0.1573` | `Expectancy 0.93%`
- `mid_3_10d`: `57` trades | `Kelly -0.3287` | `Expectancy -3.02%`
- `long_10_30d`: `26` trades | `Kelly 0.2968` | `Expectancy 4.87%`
- `very_long_30d+`: `3` trades | `Kelly 1.0000` | `Expectancy 94.89%`

## Kelly By Position Change

- `micro_<5%`: `109` trades (`63.7%` of all trades) | `Kelly 0.0437` | `Expectancy 0.38%`
- `small_5_15%`: `57` trades | `Kelly -0.0624` | `Expectancy -0.52%`
- `large_15%+`: `5` trades | `Kelly 1.0000` | `Expectancy 61.33%`

## Friction And Interpretation

- total trade actions: `578`
- rough friction estimate: `57.80%`
- removing `micro_<5%` would save about `21.80%` of friction

## What The Model Learned

The current reward is dominated by step-level portfolio return, turnover/smoothness cost, drawdown/downside/risk-budget control, and inactivity/opportunity-cost shaping.
That means the actor is mainly optimized for aggregate path quality, not for per-trade round-trip expectancy.

## Conclusion

- `mid_3_10d` is the clearest negative-expectancy bucket: `Kelly -0.3287 / Expectancy -3.02%`
- `small_5_15%` is also negative: `Kelly -0.0624 / Expectancy -0.52%`
- `micro_<5%` is near-flat but too frequent: `Kelly 0.0437 / Expectancy 0.38%`
- the actor still has portfolio-level value, but its T-style rebalances are partly a side-effect of path control rather than a clean alpha source

## Next Action Order

- raise the effective minimum rebalance threshold to suppress `micro_<5%` churn
- add or tighten `trade_lock` / `cooldown` for `3-10d` churn
- test fixed-actor low-micro replays before changing reward or architecture
