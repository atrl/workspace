# OOS Poor-Window Diagnostic: cap10_release_0.10__wf_2022

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
| 2022-09-15 | 2022-12-13 | 0.29% | 14.44% | -14.16pp | 6.81% | 0.019 | 44.16 |
| 2022-09-23 | 2022-12-21 | 0.06% | 11.07% | -11.01pp | 6.81% | 0.009 | 41.01 |
| 2022-09-30 | 2022-12-29 | -1.01% | 9.14% | -10.16pp | 8.10% | 0.001 | 40.16 |
| 2022-09-16 | 2022-12-14 | -0.06% | 7.79% | -7.86pp | 6.81% | 0.017 | 37.86 |
| 2022-09-06 | 2022-12-02 | 0.34% | 4.91% | -4.57pp | 6.79% | 0.044 | 34.57 |
| 2022-09-14 | 2022-12-12 | -0.25% | 4.17% | -4.43pp | 6.81% | 0.021 | 34.43 |
| 2022-09-02 | 2022-12-01 | 1.34% | 4.98% | -3.63pp | 6.79% | 0.048 | 33.63 |
| 2022-05-09 | 2022-08-08 | 0.61% | 4.16% | -3.55pp | 6.02% | 0.059 | 33.55 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| realized_kurtosis_21 | 0.2611 | -0.6470 | 2.023 | -0.000 |
| anchor_slope_180 | -0.1234 | -0.0125 | -1.770 | 0.044 |
| rolling_new_low_ratio_63 | 0.1548 | 0.0861 | 1.386 | -0.092 |
| tracking_error_vol_21 | 0.0792 | 0.0629 | 1.300 | -0.014 |
| negative_gap_frequency_21 | 0.5071 | 0.4196 | 1.252 | 0.087 |
| below_anchor_ratio_63 | 0.9998 | 0.5795 | 1.141 | -0.056 |
| anchor_gap_180 | -0.5207 | -0.2340 | -1.134 | -0.072 |
| vol_coef_21 | 1.2079 | 1.5020 | -1.010 | 0.012 |
| decay_63 | -0.1364 | -0.0715 | -0.953 | 0.142 |
| regime_risk_no_vix | 0.5473 | 0.6495 | -0.940 | 0.114 |
| drawdown_streak_63 | 1.0000 | 0.7438 | 0.840 | -0.024 |
| new_low_ratio_21 | 0.2256 | 0.1559 | 0.744 | -0.017 |
| trend_slope_63 | -0.1585 | -0.0711 | -0.734 | 0.119 |
| regime_risk | 0.5814 | 0.6801 | -0.701 | 0.168 |
| realized_vol_63 | 1.3044 | 1.2051 | 0.631 | -0.056 |
| vol_of_vol_21 | 0.1182 | 0.0904 | 0.620 | -0.027 |
| vol_stress_score | 0.4683 | 0.6163 | -0.612 | 0.003 |
| regime_risk_velocity_5 | -0.0208 | 0.0223 | -0.604 | 0.363 |
| vix_percentile_252 | 0.6234 | 0.7554 | -0.599 | 0.158 |
| anchor_gap_velocity_10 | 0.0110 | -0.0919 | 0.581 | -0.263 |
| vix_zscore | 0.4455 | 1.0498 | -0.483 | 0.142 |
| trend_gap_63 | -0.1578 | -0.1005 | -0.427 | 0.125 |
| tracking_error_21 | -0.0049 | -0.0031 | -0.407 | 0.171 |
| vix_momentum_10 | -0.0195 | 0.0834 | -0.407 | 0.209 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
