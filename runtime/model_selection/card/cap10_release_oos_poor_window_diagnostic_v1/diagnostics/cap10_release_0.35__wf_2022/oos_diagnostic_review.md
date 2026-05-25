# OOS Poor-Window Diagnostic: cap10_release_0.35__wf_2022

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
| 2022-09-15 | 2022-12-13 | -1.30% | 14.44% | -15.74pp | 6.80% | 0.019 | 45.74 |
| 2022-09-23 | 2022-12-21 | -1.82% | 11.07% | -12.89pp | 6.80% | 0.009 | 42.89 |
| 2022-09-30 | 2022-12-29 | -3.06% | 9.14% | -12.20pp | 8.10% | 0.002 | 42.20 |
| 2022-09-16 | 2022-12-14 | -1.92% | 7.79% | -9.71pp | 6.80% | 0.017 | 39.71 |
| 2022-05-11 | 2022-08-10 | -4.10% | 3.97% | -8.07pp | 9.65% | 0.090 | 38.07 |
| 2022-05-09 | 2022-08-08 | -2.79% | 4.16% | -6.95pp | 5.43% | 0.102 | 36.95 |
| 2022-09-06 | 2022-12-02 | -1.61% | 4.91% | -6.52pp | 6.78% | 0.044 | 36.52 |
| 2022-09-14 | 2022-12-12 | -2.22% | 4.17% | -6.39pp | 6.80% | 0.021 | 36.39 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| realized_kurtosis_21 | 0.2633 | -0.6587 | 2.116 | 0.002 |
| anchor_slope_180 | -0.1230 | -0.0120 | -1.764 | 0.048 |
| rolling_new_low_ratio_63 | 0.1548 | 0.0854 | 1.402 | -0.096 |
| tracking_error_vol_21 | 0.0791 | 0.0628 | 1.285 | -0.023 |
| negative_gap_frequency_21 | 0.5068 | 0.4190 | 1.255 | 0.078 |
| below_anchor_ratio_63 | 0.9998 | 0.5755 | 1.154 | -0.059 |
| anchor_gap_180 | -0.5196 | -0.2328 | -1.129 | -0.068 |
| vol_coef_21 | 1.2057 | 1.5078 | -1.054 | 0.012 |
| regime_risk_no_vix | 0.5460 | 0.6522 | -0.995 | 0.114 |
| decay_63 | -0.1364 | -0.0708 | -0.966 | 0.142 |
| drawdown_streak_63 | 1.0000 | 0.7413 | 0.847 | -0.026 |
| regime_risk | 0.5788 | 0.6846 | -0.776 | 0.169 |
| trend_slope_63 | -0.1585 | -0.0703 | -0.738 | 0.121 |
| new_low_ratio_21 | 0.2221 | 0.1601 | 0.670 | -0.018 |
| vix_percentile_252 | 0.6202 | 0.7611 | -0.654 | 0.158 |
| realized_vol_63 | 1.3048 | 1.2035 | 0.642 | -0.054 |
| vol_of_vol_21 | 0.1182 | 0.0901 | 0.625 | -0.031 |
| anchor_gap_velocity_10 | 0.0129 | -0.0954 | 0.613 | -0.273 |
| regime_risk_velocity_5 | -0.0210 | 0.0229 | -0.610 | 0.374 |
| vol_stress_score | 0.4710 | 0.6140 | -0.591 | 0.007 |
| vix_zscore | 0.4335 | 1.0724 | -0.515 | 0.142 |
| vix_momentum_10 | -0.0228 | 0.0890 | -0.442 | 0.215 |
| trend_gap_21 | -0.0339 | -0.0802 | 0.432 | -0.013 |
| trend_gap_63 | -0.1573 | -0.1008 | -0.418 | 0.136 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
