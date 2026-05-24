# OOS Poor-Window Diagnostic: recovery_reentry_equity_drawdown_guard_v1

## Scope

- train_end: `2018-12-31`
- OOS window: `2020-01-02` .. `2026-05-21`
- target_excess=30.00%
- target_max_dd=15.00%

## External References

- [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551): selection-bias and multiple-testing warning for repeated strategy trials
- [Probability of Backtest Overfitting](https://doi.org/10.21314/JCF.2016.322): walk-forward and OOS discipline motivation
- [Qlib GitHub](https://github.com/microsoft/qlib): keep diagnostics inside the qlib research-to-production workflow
- [Qlib arXiv](https://arxiv.org/abs/2009.11189): AI quant platform scope: data, learning, backtest, and deployment infrastructure

## Poor Windows

| start | end | strategy | buy-hold | excess | max_dd | avg_pos | bad_score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-02-05 | 2026-05-06 | 28.81% | 211.46% | -182.65pp | 4.54% | 0.180 | 187.19 |
| 2025-04-08 | 2025-07-09 | 51.79% | 223.03% | -171.24pp | 7.95% | 0.407 | 179.18 |
| 2026-02-10 | 2026-05-11 | 27.24% | 200.63% | -173.39pp | 4.54% | 0.179 | 177.93 |
| 2025-04-21 | 2025-07-21 | 49.00% | 215.73% | -166.73pp | 7.78% | 0.470 | 174.51 |
| 2026-02-12 | 2026-05-13 | 25.99% | 191.98% | -165.99pp | 4.54% | 0.179 | 170.53 |
| 2026-02-13 | 2026-05-14 | 24.24% | 188.26% | -164.03pp | 4.54% | 0.173 | 168.57 |
| 2025-04-17 | 2025-07-18 | 46.39% | 197.07% | -150.68pp | 7.78% | 0.464 | 158.46 |
| 2026-02-09 | 2026-05-08 | 23.46% | 176.34% | -152.88pp | 4.54% | 0.180 | 157.42 |
| 2026-02-23 | 2026-05-21 | 21.12% | 170.86% | -149.74pp | 7.26% | 0.140 | 157.00 |
| 2025-04-16 | 2025-07-17 | 45.82% | 192.51% | -146.68pp | 7.78% | 0.456 | 154.47 |
| 2026-02-04 | 2026-05-05 | 22.41% | 171.03% | -148.62pp | 4.54% | 0.180 | 153.15 |
| 2025-04-22 | 2025-07-22 | 43.16% | 186.71% | -143.55pp | 7.78% | 0.477 | 151.33 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| jump_variation_5 | 1.1420 | 0.1198 | 1.184 | 0.009 |
| vol_of_vol_21 | 0.2254 | 0.1164 | 1.183 | -0.022 |
| realized_vol_63 | 1.3684 | 1.0152 | 1.082 | -0.011 |
| realized_kurtosis_21 | 1.5262 | 0.2686 | 0.941 | -0.026 |
| mom_21 | 0.2731 | 0.0608 | 0.812 | -0.176 |
| downside_ratio_21 | 0.4023 | 0.4710 | -0.802 | 0.183 |
| relative_strength_21 | 1.7856 | 0.3795 | 0.802 | -0.174 |
| expectancy_proxy_21 | 0.2168 | 0.0673 | 0.801 | -0.199 |
| trend_slope_21 | 0.1023 | 0.0137 | 0.772 | 0.004 |
| sharpe_proxy_21 | 3.3285 | 1.0781 | 0.723 | -0.174 |
| vix_percentile_252 | 0.6834 | 0.4536 | 0.714 | 0.117 |
| realized_skew_21 | 0.3270 | -0.1068 | 0.700 | 0.016 |
| realized_vol_21 | 1.2760 | 1.0053 | 0.689 | 0.002 |
| trend_gap_21 | 0.0831 | 0.0072 | 0.635 | -0.020 |
| mom_10 | 0.1476 | 0.0277 | 0.634 | -0.262 |
| anchor_gap_velocity_10 | 0.1198 | -0.0024 | 0.601 | -0.241 |
| downside_ratio_10 | 0.3905 | 0.4725 | -0.583 | 0.246 |
| rsi_14 | 0.6131 | 0.5261 | 0.564 | -0.189 |
| gain_streak_10 | 0.1850 | 0.1058 | 0.545 | -0.782 |
| realized_vol_10 | 1.2198 | 0.9934 | 0.509 | 0.001 |
| mom_63 | 0.3978 | 0.2033 | 0.481 | -0.090 |
| mom_5 | 0.0774 | 0.0133 | 0.461 | -0.362 |
| negative_gap_frequency_21 | 0.3984 | 0.3439 | 0.453 | 0.086 |
| anchor_gap_180 | 0.3181 | 0.1346 | 0.419 | -0.110 |
| rsi_6 | 0.6251 | 0.5306 | 0.411 | -0.573 |
| rolling_new_low_ratio_63 | 0.0651 | 0.0419 | 0.398 | -0.006 |
| vix_zscore | 0.4439 | -0.0645 | 0.376 | 0.109 |
| decay_63 | -0.3349 | -0.2275 | -0.373 | 0.063 |
| trend_gap_63 | 0.0842 | 0.0224 | 0.366 | 0.017 |
| new_low_ratio_21 | 0.0541 | 0.0920 | -0.319 | 0.011 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
