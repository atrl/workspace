# OOS Poor-Window Diagnostic: cap10_release_0.20__wf_2022

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
| 2022-09-15 | 2022-12-13 | 8.73% | 14.44% | -5.71pp | 6.75% | 0.019 | 35.71 |
| 2022-05-19 | 2022-08-18 | -1.55% | 1.85% | -3.40pp | 6.30% | 0.078 | 33.40 |
| 2022-05-09 | 2022-08-08 | 0.78% | 4.16% | -3.38pp | 6.30% | 0.065 | 33.38 |
| 2022-05-11 | 2022-08-10 | 0.77% | 3.97% | -3.20pp | 6.30% | 0.064 | 33.20 |
| 2022-05-16 | 2022-08-15 | -1.12% | 1.92% | -3.04pp | 6.30% | 0.071 | 33.04 |
| 2022-09-23 | 2022-12-21 | 8.50% | 11.07% | -2.57pp | 6.75% | 0.009 | 32.57 |
| 2022-09-30 | 2022-12-29 | 7.34% | 9.14% | -1.81pp | 8.06% | 0.002 | 31.81 |
| 2022-05-12 | 2022-08-11 | 0.57% | 1.91% | -1.34pp | 6.30% | 0.066 | 31.34 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| realized_kurtosis_21 | 0.2638 | -0.6507 | 2.074 | 0.004 |
| anchor_slope_180 | -0.1213 | -0.0154 | -1.557 | 0.043 |
| negative_gap_frequency_21 | 0.5120 | 0.4128 | 1.510 | 0.085 |
| rolling_new_low_ratio_63 | 0.1564 | 0.0839 | 1.482 | -0.093 |
| tracking_error_vol_21 | 0.0787 | 0.0635 | 1.283 | -0.017 |
| below_anchor_ratio_63 | 0.9998 | 0.5795 | 1.141 | -0.056 |
| vol_coef_21 | 1.2055 | 1.5052 | -1.048 | 0.013 |
| regime_risk | 0.5717 | 0.6934 | -1.025 | 0.168 |
| anchor_gap_180 | -0.5129 | -0.2446 | -1.019 | -0.070 |
| regime_risk_no_vix | 0.5451 | 0.6524 | -1.014 | 0.116 |
| vix_percentile_252 | 0.6083 | 0.7759 | -0.872 | 0.158 |
| drawdown_streak_63 | 1.0000 | 0.7438 | 0.840 | -0.024 |
| decay_63 | -0.1336 | -0.0752 | -0.801 | 0.141 |
| mom_21 | -0.0095 | -0.1694 | 0.749 | -0.207 |
| relative_strength_21 | -0.0805 | -1.1401 | 0.745 | -0.206 |
| trend_slope_63 | -0.1581 | -0.0716 | -0.726 | 0.119 |
| trend_gap_21 | -0.0221 | -0.0960 | 0.701 | -0.008 |
| sharpe_proxy_21 | -0.1715 | -1.6931 | 0.700 | -0.199 |
| anchor_gap_velocity_10 | 0.0180 | -0.1015 | 0.693 | -0.266 |
| realized_vol_63 | 1.3072 | 1.2012 | 0.678 | -0.056 |
| vol_of_vol_21 | 0.1192 | 0.0889 | 0.671 | -0.027 |
| rsi_14 | 0.4785 | 0.4056 | 0.623 | -0.222 |
| expectancy_proxy_21 | -0.0190 | -0.1001 | 0.612 | -0.225 |
| vix_zscore | 0.3957 | 1.1180 | -0.610 | 0.141 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
