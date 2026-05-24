# OOS Poor-Window Diagnostic: recovery_reentry_floor_v1

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
| 2026-02-10 | 2026-05-11 | 33.78% | 200.63% | -166.85pp | 6.42% | 0.229 | 173.27 |
| 2026-02-05 | 2026-05-06 | 46.84% | 211.46% | -164.61pp | 6.42% | 0.252 | 171.03 |
| 2026-02-12 | 2026-05-13 | 32.77% | 191.98% | -159.21pp | 6.42% | 0.213 | 165.63 |
| 2026-02-13 | 2026-05-14 | 30.70% | 188.26% | -157.56pp | 6.42% | 0.206 | 163.98 |
| 2025-04-08 | 2025-07-09 | 82.77% | 223.03% | -140.26pp | 16.32% | 0.552 | 156.59 |
| 2025-04-21 | 2025-07-21 | 73.80% | 215.73% | -141.93pp | 12.48% | 0.582 | 154.40 |
| 2026-02-09 | 2026-05-08 | 29.14% | 176.34% | -147.20pp | 6.42% | 0.236 | 153.62 |
| 2026-02-23 | 2026-05-21 | 26.47% | 170.86% | -144.40pp | 7.26% | 0.173 | 151.66 |
| 2026-02-20 | 2026-05-20 | 23.99% | 158.08% | -134.10pp | 7.26% | 0.175 | 141.36 |
| 2025-04-17 | 2025-07-18 | 68.55% | 197.07% | -128.52pp | 12.48% | 0.580 | 141.00 |
| 2026-02-11 | 2026-05-12 | 21.86% | 153.52% | -131.66pp | 6.42% | 0.221 | 138.08 |
| 2026-02-04 | 2026-05-05 | 39.63% | 171.03% | -131.40pp | 6.42% | 0.259 | 137.82 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| vol_of_vol_21 | 0.2264 | 0.1163 | 1.194 | -0.021 |
| jump_variation_5 | 1.1499 | 0.1197 | 1.194 | 0.009 |
| realized_vol_63 | 1.3720 | 1.0151 | 1.093 | -0.011 |
| realized_kurtosis_21 | 1.5223 | 0.2699 | 0.937 | -0.034 |
| mom_21 | 0.2734 | 0.0609 | 0.813 | -0.187 |
| relative_strength_21 | 1.7876 | 0.3802 | 0.803 | -0.185 |
| downside_ratio_21 | 0.4028 | 0.4709 | -0.795 | 0.189 |
| expectancy_proxy_21 | 0.2159 | 0.0675 | 0.795 | -0.210 |
| trend_slope_21 | 0.1022 | 0.0137 | 0.770 | -0.007 |
| vix_percentile_252 | 0.6859 | 0.4535 | 0.722 | 0.116 |
| sharpe_proxy_21 | 3.3180 | 1.0807 | 0.719 | -0.187 |
| realized_vol_21 | 1.2809 | 1.0050 | 0.702 | 0.003 |
| realized_skew_21 | 0.3242 | -0.1063 | 0.694 | 0.006 |
| mom_10 | 0.1486 | 0.0277 | 0.640 | -0.266 |
| trend_gap_21 | 0.0834 | 0.0072 | 0.638 | -0.031 |
| anchor_gap_velocity_10 | 0.1205 | -0.0024 | 0.605 | -0.247 |
| downside_ratio_10 | 0.3904 | 0.4724 | -0.584 | 0.245 |
| rsi_14 | 0.6133 | 0.5262 | 0.565 | -0.199 |
| gain_streak_10 | 0.1863 | 0.1058 | 0.555 | -0.783 |
| realized_vol_10 | 1.2257 | 0.9929 | 0.523 | 0.007 |
| mom_5 | 0.0781 | 0.0133 | 0.466 | -0.364 |
| mom_63 | 0.3923 | 0.2040 | 0.465 | -0.096 |
| negative_gap_frequency_21 | 0.3992 | 0.3438 | 0.460 | 0.093 |
| anchor_gap_180 | 0.3194 | 0.1346 | 0.422 | -0.112 |
| rsi_6 | 0.6276 | 0.5304 | 0.422 | -0.576 |
| rolling_new_low_ratio_63 | 0.0656 | 0.0418 | 0.406 | -0.006 |
| vix_zscore | 0.4506 | -0.0649 | 0.381 | 0.107 |
| decay_63 | -0.3354 | -0.2275 | -0.375 | 0.063 |
| trend_gap_63 | 0.0826 | 0.0226 | 0.355 | 0.010 |
| vol_coef_21 | 1.2726 | 1.1130 | 0.317 | 0.002 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
