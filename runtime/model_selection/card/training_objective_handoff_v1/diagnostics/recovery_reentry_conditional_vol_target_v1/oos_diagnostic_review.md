# OOS Poor-Window Diagnostic: recovery_reentry_conditional_vol_target_v1

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
| 2026-02-10 | 2026-05-11 | 25.60% | 200.63% | -175.03pp | 6.42% | 0.212 | 181.45 |
| 2026-02-05 | 2026-05-06 | 37.86% | 211.46% | -173.59pp | 6.42% | 0.235 | 180.01 |
| 2026-02-12 | 2026-05-13 | 24.65% | 191.98% | -167.33pp | 6.42% | 0.197 | 173.75 |
| 2026-02-13 | 2026-05-14 | 22.71% | 188.26% | -165.55pp | 6.42% | 0.189 | 171.98 |
| 2025-04-08 | 2025-07-09 | 64.55% | 223.03% | -158.48pp | 11.50% | 0.468 | 169.98 |
| 2025-04-21 | 2025-07-21 | 61.31% | 215.73% | -154.42pp | 11.50% | 0.531 | 165.91 |
| 2026-02-09 | 2026-05-08 | 21.24% | 176.34% | -155.09pp | 6.42% | 0.220 | 161.52 |
| 2026-02-23 | 2026-05-21 | 18.73% | 170.86% | -152.13pp | 7.26% | 0.156 | 159.39 |
| 2025-04-17 | 2025-07-18 | 58.40% | 197.07% | -138.66pp | 11.50% | 0.525 | 150.16 |
| 2026-02-20 | 2026-05-20 | 16.40% | 158.08% | -141.68pp | 7.26% | 0.159 | 148.94 |
| 2026-02-04 | 2026-05-05 | 31.09% | 171.03% | -139.94pp | 6.42% | 0.243 | 146.36 |
| 2025-04-16 | 2025-07-17 | 57.76% | 192.51% | -134.74pp | 11.50% | 0.517 | 146.24 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| vol_of_vol_21 | 0.2264 | 0.1163 | 1.194 | -0.024 |
| jump_variation_5 | 1.1499 | 0.1197 | 1.194 | 0.006 |
| realized_vol_63 | 1.3720 | 1.0151 | 1.093 | -0.009 |
| realized_kurtosis_21 | 1.5223 | 0.2699 | 0.937 | -0.033 |
| mom_21 | 0.2734 | 0.0609 | 0.813 | -0.186 |
| relative_strength_21 | 1.7876 | 0.3802 | 0.803 | -0.185 |
| downside_ratio_21 | 0.4028 | 0.4709 | -0.795 | 0.189 |
| expectancy_proxy_21 | 0.2159 | 0.0675 | 0.795 | -0.209 |
| trend_slope_21 | 0.1022 | 0.0137 | 0.770 | -0.005 |
| vix_percentile_252 | 0.6859 | 0.4535 | 0.722 | 0.114 |
| sharpe_proxy_21 | 3.3180 | 1.0807 | 0.719 | -0.185 |
| realized_vol_21 | 1.2809 | 1.0050 | 0.702 | -0.002 |
| realized_skew_21 | 0.3242 | -0.1063 | 0.694 | 0.008 |
| mom_10 | 0.1486 | 0.0277 | 0.640 | -0.267 |
| trend_gap_21 | 0.0834 | 0.0072 | 0.638 | -0.031 |
| anchor_gap_velocity_10 | 0.1205 | -0.0024 | 0.605 | -0.248 |
| downside_ratio_10 | 0.3904 | 0.4724 | -0.584 | 0.247 |
| rsi_14 | 0.6133 | 0.5262 | 0.565 | -0.198 |
| gain_streak_10 | 0.1863 | 0.1058 | 0.555 | -0.781 |
| realized_vol_10 | 1.2257 | 0.9929 | 0.523 | 0.004 |
| mom_5 | 0.0781 | 0.0133 | 0.466 | -0.365 |
| mom_63 | 0.3923 | 0.2040 | 0.465 | -0.100 |
| negative_gap_frequency_21 | 0.3992 | 0.3438 | 0.460 | 0.093 |
| anchor_gap_180 | 0.3194 | 0.1346 | 0.422 | -0.116 |
| rsi_6 | 0.6276 | 0.5304 | 0.422 | -0.574 |
| rolling_new_low_ratio_63 | 0.0656 | 0.0418 | 0.406 | -0.003 |
| vix_zscore | 0.4506 | -0.0649 | 0.381 | 0.104 |
| decay_63 | -0.3354 | -0.2275 | -0.375 | 0.064 |
| trend_gap_63 | 0.0826 | 0.0226 | 0.355 | 0.008 |
| vol_coef_21 | 1.2726 | 1.1130 | 0.317 | -0.003 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
