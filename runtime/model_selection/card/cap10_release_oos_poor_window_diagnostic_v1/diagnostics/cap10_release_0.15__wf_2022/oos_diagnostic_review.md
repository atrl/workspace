# OOS Poor-Window Diagnostic: cap10_release_0.15__wf_2022

## Scope

- train_end: `2020-12-31`
- OOS window: `2022-01-01` .. `2022-12-31`
- target_excess=30.00%
- target_max_dd=15.00%

## External References

- [Qlib GitHub](https://github.com/microsoft/qlib): keep the replay/backtest workflow inside the qlib harness
- [Qlib arXiv](https://arxiv.org/abs/2009.11189): AI quant platform scope for data, modeling, backtest, and deployment
- [Probability of Backtest Overfitting](https://doi.org/10.21314/JCF.2016.322): avoid selecting a candidate from repeated screens without OOS discipline
- [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551): multiple-testing caution for model and factor screens

## Poor Windows

| start | end | strategy | buy-hold | excess | max_dd | avg_pos | bad_score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-09-15 | 2022-12-13 | 5.23% | 14.44% | -9.21pp | 4.29% | 0.019 | 39.21 |
| 2022-09-23 | 2022-12-21 | 4.99% | 11.07% | -6.08pp | 5.34% | 0.009 | 36.08 |
| 2022-09-30 | 2022-12-29 | 3.86% | 9.14% | -5.28pp | 6.73% | 0.002 | 35.28 |
| 2022-09-16 | 2022-12-14 | 4.86% | 7.79% | -2.93pp | 4.29% | 0.017 | 32.93 |
| 2022-05-09 | 2022-08-08 | 4.48% | 4.16% | 0.32pp | 4.57% | 0.070 | 29.68 |
| 2022-05-11 | 2022-08-10 | 4.45% | 3.97% | 0.48pp | 4.57% | 0.068 | 29.52 |
| 2022-09-14 | 2022-12-12 | 4.66% | 4.17% | 0.49pp | 4.29% | 0.021 | 29.51 |
| 2022-09-06 | 2022-12-02 | 6.46% | 4.91% | 1.55pp | 4.04% | 0.041 | 28.45 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| realized_kurtosis_21 | 0.2633 | -0.6587 | 2.116 | 0.004 |
| anchor_slope_180 | -0.1230 | -0.0120 | -1.764 | 0.045 |
| rolling_new_low_ratio_63 | 0.1548 | 0.0854 | 1.402 | -0.092 |
| tracking_error_vol_21 | 0.0791 | 0.0628 | 1.285 | -0.016 |
| negative_gap_frequency_21 | 0.5068 | 0.4190 | 1.255 | 0.087 |
| below_anchor_ratio_63 | 0.9998 | 0.5755 | 1.154 | -0.056 |
| anchor_gap_180 | -0.5196 | -0.2328 | -1.129 | -0.068 |
| vol_coef_21 | 1.2057 | 1.5078 | -1.054 | 0.011 |
| regime_risk_no_vix | 0.5460 | 0.6522 | -0.995 | 0.111 |
| decay_63 | -0.1364 | -0.0708 | -0.966 | 0.139 |
| drawdown_streak_63 | 1.0000 | 0.7413 | 0.847 | -0.024 |
| regime_risk | 0.5788 | 0.6846 | -0.776 | 0.164 |
| trend_slope_63 | -0.1585 | -0.0703 | -0.738 | 0.118 |
| new_low_ratio_21 | 0.2221 | 0.1601 | 0.670 | -0.022 |
| vix_percentile_252 | 0.6202 | 0.7611 | -0.654 | 0.154 |
| realized_vol_63 | 1.3048 | 1.2035 | 0.642 | -0.054 |
| vol_of_vol_21 | 0.1182 | 0.0901 | 0.625 | -0.023 |
| anchor_gap_velocity_10 | 0.0129 | -0.0954 | 0.613 | -0.263 |
| regime_risk_velocity_5 | -0.0210 | 0.0229 | -0.610 | 0.365 |
| vol_stress_score | 0.4710 | 0.6140 | -0.591 | 0.005 |
| vix_zscore | 0.4335 | 1.0724 | -0.515 | 0.138 |
| vix_momentum_10 | -0.0228 | 0.0890 | -0.442 | 0.208 |
| trend_gap_21 | -0.0339 | -0.0802 | 0.432 | -0.004 |
| trend_gap_63 | -0.1573 | -0.1008 | -0.418 | 0.132 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
