# OOS Poor-Window Diagnostic: soxl_soxx_efficiency_chop_guard_v1

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
| 2026-02-05 | 2026-05-06 | 27.99% | 211.46% | -183.46pp | 6.42% | 0.183 | 189.88 |
| 2026-02-10 | 2026-05-11 | 23.32% | 200.63% | -177.31pp | 6.42% | 0.175 | 183.74 |
| 2026-02-12 | 2026-05-13 | 22.16% | 191.98% | -169.82pp | 6.42% | 0.171 | 176.24 |
| 2026-02-13 | 2026-05-14 | 21.18% | 188.26% | -167.08pp | 6.42% | 0.168 | 173.50 |
| 2025-04-08 | 2025-07-09 | 63.77% | 223.03% | -159.26pp | 11.50% | 0.466 | 170.76 |
| 2025-04-21 | 2025-07-21 | 60.76% | 215.73% | -154.97pp | 11.50% | 0.530 | 166.47 |
| 2026-02-09 | 2026-05-08 | 19.47% | 176.34% | -156.87pp | 6.42% | 0.178 | 163.29 |
| 2026-02-23 | 2026-05-21 | 18.73% | 170.86% | -152.13pp | 7.26% | 0.156 | 159.39 |
| 2026-02-04 | 2026-05-05 | 21.66% | 171.03% | -149.37pp | 6.42% | 0.185 | 155.79 |
| 2025-04-17 | 2025-07-18 | 57.94% | 197.07% | -139.12pp | 11.50% | 0.524 | 150.62 |
| 2026-02-20 | 2026-05-20 | 16.40% | 158.08% | -141.68pp | 7.26% | 0.159 | 148.94 |
| 2025-04-16 | 2025-07-17 | 57.33% | 192.51% | -135.18pp | 11.50% | 0.515 | 146.67 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| vol_of_vol_21 | 0.2264 | 0.1163 | 1.194 | -0.027 |
| jump_variation_5 | 1.1499 | 0.1197 | 1.194 | 0.007 |
| realized_vol_63 | 1.3720 | 1.0151 | 1.093 | -0.008 |
| realized_kurtosis_21 | 1.5223 | 0.2699 | 0.937 | -0.031 |
| mom_21 | 0.2734 | 0.0609 | 0.813 | -0.185 |
| relative_strength_21 | 1.7876 | 0.3802 | 0.803 | -0.184 |
| downside_ratio_21 | 0.4028 | 0.4709 | -0.795 | 0.187 |
| expectancy_proxy_21 | 0.2159 | 0.0675 | 0.795 | -0.207 |
| trend_slope_21 | 0.1022 | 0.0137 | 0.770 | -0.005 |
| vix_percentile_252 | 0.6859 | 0.4535 | 0.722 | 0.119 |
| sharpe_proxy_21 | 3.3180 | 1.0807 | 0.719 | -0.183 |
| realized_vol_21 | 1.2809 | 1.0050 | 0.702 | 0.000 |
| realized_skew_21 | 0.3242 | -0.1063 | 0.694 | 0.010 |
| mom_10 | 0.1486 | 0.0277 | 0.640 | -0.266 |
| trend_gap_21 | 0.0834 | 0.0072 | 0.638 | -0.025 |
| anchor_gap_velocity_10 | 0.1205 | -0.0024 | 0.605 | -0.245 |
| downside_ratio_10 | 0.3904 | 0.4724 | -0.584 | 0.244 |
| rsi_14 | 0.6133 | 0.5262 | 0.565 | -0.193 |
| gain_streak_10 | 0.1863 | 0.1058 | 0.555 | -0.783 |
| realized_vol_10 | 1.2257 | 0.9929 | 0.523 | -0.001 |
| mom_5 | 0.0781 | 0.0133 | 0.466 | -0.367 |
| mom_63 | 0.3923 | 0.2040 | 0.465 | -0.094 |
| negative_gap_frequency_21 | 0.3992 | 0.3438 | 0.460 | 0.088 |
| anchor_gap_180 | 0.3194 | 0.1346 | 0.422 | -0.113 |
| rsi_6 | 0.6276 | 0.5304 | 0.422 | -0.579 |
| rolling_new_low_ratio_63 | 0.0656 | 0.0418 | 0.406 | -0.009 |
| vix_zscore | 0.4506 | -0.0649 | 0.381 | 0.111 |
| decay_63 | -0.3354 | -0.2275 | -0.375 | 0.066 |
| trend_gap_63 | 0.0826 | 0.0226 | 0.355 | 0.011 |
| vol_coef_21 | 1.2726 | 1.1130 | 0.317 | -0.000 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
