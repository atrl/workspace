# OOS Poor-Window Diagnostic: short_volshock_plain__wf_2022

## Scope

- train_end: `fold-local replay source`
- OOS window: `2022-01-03` .. `2022-12-29`
- target_excess=30.00%
- target_max_dd=15.00%

## External References

- [Qlib](https://github.com/microsoft/qlib): reproducible quant workflow and artifact discipline
- [Qlib arXiv](https://arxiv.org/abs/2009.11189): AI-oriented quantitative investment platform
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL): DRL trading workflow with train/validate/test separation
- [DQfD](https://arxiv.org/abs/1704.03732): demonstration learning still needs controlled policy improvement
- [Risk-sensitive RL](https://arxiv.org/abs/1312.4077): risk-sensitive sequential decision objective basis
- [PBO](https://doi.org/10.21314/JCF.2016.322): repeated trials require OOS and walk-forward discipline

## Poor Windows

| start | end | strategy | buy-hold | excess | max_dd | avg_pos | bad_score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022-10-14 | 2022-12-13 | 8.47% | 96.68% | -88.21pp | 5.92% | 0.156 | 118.21 |
| 2022-10-17 | 2022-12-14 | 5.65% | 77.35% | -71.70pp | 5.92% | 0.157 | 101.70 |
| 2022-10-12 | 2022-12-09 | 5.88% | 67.85% | -61.97pp | 5.92% | 0.149 | 91.97 |
| 2022-10-11 | 2022-12-08 | 6.16% | 66.71% | -60.55pp | 5.92% | 0.146 | 90.55 |
| 2022-10-13 | 2022-12-12 | 6.35% | 62.25% | -55.90pp | 5.92% | 0.152 | 85.90 |
| 2022-06-16 | 2022-08-16 | 1.52% | 55.78% | -54.27pp | 9.12% | 0.148 | 84.27 |
| 2022-10-18 | 2022-12-15 | 4.27% | 54.05% | -49.78pp | 6.38% | 0.157 | 79.78 |
| 2022-06-17 | 2022-08-17 | -1.91% | 42.70% | -44.61pp | 9.12% | 0.157 | 74.61 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| expectancy_proxy_21 | 0.0575 | -0.1135 | 1.281 | -0.229 |
| mom_21 | 0.0887 | -0.1687 | 1.213 | -0.211 |
| relative_strength_21 | 0.5712 | -1.1361 | 1.208 | -0.210 |
| rolling_new_low_ratio_63 | 0.1674 | 0.1028 | 1.173 | -0.094 |
| sharpe_proxy_21 | 0.8333 | -1.7199 | 1.160 | -0.204 |
| vix_percentile_252 | 0.5368 | 0.7622 | -1.061 | 0.161 |
| trend_gap_21 | 0.0149 | -0.0909 | 1.061 | -0.013 |
| trend_slope_21 | 0.0018 | -0.0891 | 1.048 | 0.072 |
| regime_risk | 0.5344 | 0.6751 | -1.036 | 0.172 |
| rsi_14 | 0.5223 | 0.4064 | 1.022 | -0.225 |
| downside_ratio_21 | 0.4767 | 0.5461 | -0.986 | 0.231 |
| anchor_slope_180 | -0.1248 | -0.0489 | -0.935 | 0.045 |
| vol_coef_21 | 1.1348 | 1.4469 | -0.888 | 0.016 |
| regime_risk_velocity_5 | -0.0414 | 0.0195 | -0.846 | 0.366 |
| realized_kurtosis_21 | 0.3405 | -0.3829 | 0.844 | 0.003 |
| downside_semivol_21 | 0.8320 | 0.9501 | -0.826 | 0.108 |
| vix_zscore | 0.1204 | 1.0373 | -0.824 | 0.143 |
| mom_10 | 0.0595 | -0.0982 | 0.820 | -0.280 |
| anchor_gap_velocity_10 | 0.0471 | -0.0770 | 0.813 | -0.266 |
| days_since_high_63 | 0.8707 | 0.6726 | 0.793 | -0.100 |
| trend_slope_63 | -0.1757 | -0.0918 | -0.790 | 0.113 |
| regime_risk_no_vix | 0.5275 | 0.6279 | -0.777 | 0.116 |
| below_anchor_ratio_63 | 1.0000 | 0.7212 | 0.775 | -0.060 |
| drawdown_21 | -0.1850 | -0.2794 | 0.761 | -0.293 |
| cum_drop_20 | -0.1791 | -0.2722 | 0.756 | -0.301 |
| cum_drop_30 | -0.2354 | -0.3345 | 0.752 | -0.199 |
| rsi_6 | 0.5411 | 0.3944 | 0.723 | -0.632 |
| tracking_error_vol_21 | 0.0795 | 0.0682 | 0.678 | -0.014 |
| downside_ratio_10 | 0.4756 | 0.5525 | -0.659 | 0.236 |
| anchor_gap_180 | -0.5068 | -0.3395 | -0.649 | -0.069 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
