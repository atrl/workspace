# OOS Poor-Window Diagnostic: v19reg_v22_budget_usage_bull_guard_v1_current

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
| 2026-02-10 | 2026-05-11 | 1.74% | 200.63% | -198.89pp | 6.42% | 0.140 | 205.31 |
| 2026-02-05 | 2026-05-06 | 17.69% | 211.46% | -193.77pp | 6.42% | 0.179 | 200.19 |
| 2026-02-12 | 2026-05-13 | 1.99% | 191.98% | -190.00pp | 6.42% | 0.113 | 196.42 |
| 2026-02-13 | 2026-05-14 | 0.02% | 188.26% | -188.24pp | 6.42% | 0.100 | 194.66 |
| 2026-02-09 | 2026-05-08 | 0.83% | 176.34% | -175.51pp | 6.42% | 0.153 | 181.93 |
| 2026-02-23 | 2026-05-21 | -2.20% | 170.86% | -173.07pp | 6.42% | 0.040 | 179.49 |
| 2026-02-20 | 2026-05-20 | -3.12% | 158.08% | -161.20pp | 6.42% | 0.048 | 167.62 |
| 2026-02-11 | 2026-05-12 | -4.17% | 153.52% | -157.69pp | 6.42% | 0.127 | 164.11 |
| 2026-02-17 | 2026-05-15 | 0.33% | 155.14% | -154.80pp | 6.42% | 0.087 | 161.22 |
| 2026-02-04 | 2026-05-05 | 17.80% | 171.03% | -153.23pp | 6.42% | 0.193 | 159.65 |
| 2025-04-08 | 2025-07-09 | 82.77% | 223.03% | -140.26pp | 16.32% | 0.552 | 156.59 |
| 2025-04-21 | 2025-07-21 | 73.80% | 215.73% | -141.93pp | 12.48% | 0.582 | 154.40 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| vol_of_vol_21 | 0.2264 | 0.1163 | 1.194 | -0.022 |
| jump_variation_5 | 1.1499 | 0.1197 | 1.194 | 0.009 |
| realized_vol_63 | 1.3720 | 1.0151 | 1.093 | -0.011 |
| realized_kurtosis_21 | 1.5223 | 0.2699 | 0.937 | -0.035 |
| mom_21 | 0.2734 | 0.0609 | 0.813 | -0.190 |
| relative_strength_21 | 1.7876 | 0.3802 | 0.803 | -0.188 |
| downside_ratio_21 | 0.4028 | 0.4709 | -0.795 | 0.192 |
| expectancy_proxy_21 | 0.2159 | 0.0675 | 0.795 | -0.213 |
| trend_slope_21 | 0.1022 | 0.0137 | 0.770 | -0.010 |
| vix_percentile_252 | 0.6859 | 0.4535 | 0.722 | 0.116 |
| sharpe_proxy_21 | 3.3180 | 1.0807 | 0.719 | -0.189 |
| realized_vol_21 | 1.2809 | 1.0050 | 0.702 | 0.002 |
| realized_skew_21 | 0.3242 | -0.1063 | 0.694 | 0.006 |
| mom_10 | 0.1486 | 0.0277 | 0.640 | -0.269 |
| trend_gap_21 | 0.0834 | 0.0072 | 0.638 | -0.034 |
| anchor_gap_velocity_10 | 0.1205 | -0.0024 | 0.605 | -0.249 |
| downside_ratio_10 | 0.3904 | 0.4724 | -0.584 | 0.248 |
| rsi_14 | 0.6133 | 0.5262 | 0.565 | -0.202 |
| gain_streak_10 | 0.1863 | 0.1058 | 0.555 | -0.784 |
| realized_vol_10 | 1.2257 | 0.9929 | 0.523 | 0.007 |
| mom_5 | 0.0781 | 0.0133 | 0.466 | -0.365 |
| mom_63 | 0.3923 | 0.2040 | 0.465 | -0.099 |
| negative_gap_frequency_21 | 0.3992 | 0.3438 | 0.460 | 0.094 |
| anchor_gap_180 | 0.3194 | 0.1346 | 0.422 | -0.116 |
| rsi_6 | 0.6276 | 0.5304 | 0.422 | -0.577 |
| rolling_new_low_ratio_63 | 0.0656 | 0.0418 | 0.406 | -0.005 |
| vix_zscore | 0.4506 | -0.0649 | 0.381 | 0.107 |
| decay_63 | -0.3354 | -0.2275 | -0.375 | 0.064 |
| trend_gap_63 | 0.0826 | 0.0226 | 0.355 | 0.009 |
| vol_coef_21 | 1.2726 | 1.1130 | 0.317 | 0.001 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
