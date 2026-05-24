# OOS Poor-Window Diagnostic: recovery_reentry_cppi_floor_v1

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
| 2025-04-08 | 2025-07-09 | 38.93% | 223.03% | -184.10pp | 11.50% | 0.385 | 195.60 |
| 2025-04-21 | 2025-07-21 | 42.04% | 215.73% | -173.69pp | 11.50% | 0.479 | 185.19 |
| 2026-02-10 | 2026-05-11 | 25.60% | 200.63% | -175.03pp | 6.42% | 0.212 | 181.45 |
| 2026-02-05 | 2026-05-06 | 37.86% | 211.46% | -173.59pp | 6.42% | 0.235 | 180.01 |
| 2026-02-12 | 2026-05-13 | 24.65% | 191.98% | -167.33pp | 6.42% | 0.197 | 173.75 |
| 2026-02-13 | 2026-05-14 | 22.71% | 188.26% | -165.55pp | 6.42% | 0.189 | 171.98 |
| 2025-04-17 | 2025-07-18 | 41.36% | 197.07% | -155.71pp | 11.50% | 0.469 | 167.21 |
| 2025-04-04 | 2025-07-07 | 32.93% | 186.03% | -153.10pp | 11.50% | 0.364 | 164.60 |
| 2025-04-16 | 2025-07-17 | 41.32% | 192.51% | -151.19pp | 11.50% | 0.457 | 162.69 |
| 2025-04-07 | 2025-07-08 | 37.94% | 188.20% | -150.25pp | 11.50% | 0.374 | 161.75 |
| 2026-02-09 | 2026-05-08 | 21.24% | 176.34% | -155.09pp | 6.42% | 0.220 | 161.52 |
| 2025-04-22 | 2025-07-22 | 38.24% | 186.71% | -148.47pp | 11.50% | 0.489 | 159.97 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| jump_variation_5 | 1.2984 | 0.1073 | 1.492 | 0.005 |
| vol_of_vol_21 | 0.2280 | 0.1164 | 1.212 | -0.023 |
| realized_vol_63 | 1.3729 | 1.0158 | 1.095 | -0.013 |
| realized_kurtosis_21 | 1.6351 | 0.2614 | 1.028 | -0.035 |
| vix_percentile_252 | 0.6920 | 0.4534 | 0.743 | 0.111 |
| downside_ratio_21 | 0.4079 | 0.4702 | -0.727 | 0.188 |
| expectancy_proxy_21 | 0.2043 | 0.0689 | 0.726 | -0.207 |
| mom_21 | 0.2489 | 0.0638 | 0.708 | -0.183 |
| realized_skew_21 | 0.3301 | -0.1060 | 0.706 | 0.005 |
| relative_strength_21 | 1.6229 | 0.3992 | 0.697 | -0.181 |
| realized_vol_21 | 1.2769 | 1.0060 | 0.690 | -0.001 |
| sharpe_proxy_21 | 3.1557 | 1.1011 | 0.661 | -0.183 |
| anchor_gap_velocity_10 | 0.1228 | -0.0024 | 0.615 | -0.243 |
| trend_slope_21 | 0.0869 | 0.0154 | 0.611 | -0.003 |
| mom_10 | 0.1365 | 0.0291 | 0.574 | -0.262 |
| trend_gap_21 | 0.0752 | 0.0082 | 0.562 | -0.027 |
| gain_streak_10 | 0.1867 | 0.1059 | 0.557 | -0.783 |
| downside_ratio_10 | 0.3944 | 0.4719 | -0.552 | 0.243 |
| rsi_14 | 0.6076 | 0.5269 | 0.525 | -0.195 |
| negative_gap_frequency_21 | 0.4059 | 0.3433 | 0.522 | 0.089 |
| realized_vol_10 | 1.2144 | 0.9945 | 0.495 | 0.005 |
| vix_zscore | 0.5550 | -0.0740 | 0.473 | 0.102 |
| rolling_new_low_ratio_63 | 0.0688 | 0.0416 | 0.469 | -0.009 |
| mom_5 | 0.0775 | 0.0134 | 0.464 | -0.362 |
| rsi_6 | 0.6265 | 0.5308 | 0.417 | -0.572 |
| decay_63 | -0.3416 | -0.2271 | -0.399 | 0.064 |
| mom_63 | 0.3551 | 0.2080 | 0.363 | -0.092 |
| below_anchor_ratio_63 | 0.5001 | 0.3667 | 0.325 | 0.009 |
| vol_coef_21 | 1.2707 | 1.1135 | 0.313 | -0.001 |
| new_low_ratio_21 | 0.0603 | 0.0913 | -0.262 | 0.019 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
