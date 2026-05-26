# OOS Poor-Window Diagnostic: short_volshock_weighted_obj025_cb099__wf_2023_2024

## Scope

- train_end: `fold-local replay source`
- OOS window: `2024-01-02` .. `2024-12-30`
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
| 2024-04-19 | 2024-06-18 | 12.25% | 111.89% | -99.64pp | 2.82% | 0.150 | 129.64 |
| 2024-01-05 | 2024-03-06 | 10.99% | 93.68% | -82.69pp | 6.07% | 0.232 | 112.69 |
| 2024-01-08 | 2024-03-07 | 14.09% | 94.58% | -80.49pp | 6.07% | 0.239 | 110.49 |
| 2024-01-03 | 2024-03-04 | 10.16% | 89.60% | -79.44pp | 6.07% | 0.220 | 109.44 |
| 2024-04-22 | 2024-06-20 | 11.04% | 87.15% | -76.12pp | 2.82% | 0.151 | 106.12 |
| 2024-01-04 | 2024-03-05 | 7.55% | 83.11% | -75.57pp | 6.07% | 0.226 | 105.57 |
| 2024-04-18 | 2024-06-17 | 10.21% | 79.52% | -69.31pp | 2.82% | 0.151 | 99.31 |
| 2024-01-11 | 2024-03-12 | 5.27% | 74.21% | -68.93pp | 8.24% | 0.232 | 98.93 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| negative_gap_frequency_21 | 0.2474 | 0.3633 | -1.393 | 0.109 |
| drawdown_63 | -0.1185 | -0.3164 | 1.343 | -0.182 |
| mom_63 | 0.4767 | 0.0815 | 1.313 | -0.081 |
| anchor_gap_180 | 0.5084 | 0.0250 | 1.313 | -0.162 |
| vix_percentile_252 | 0.2771 | 0.6494 | -1.310 | 0.222 |
| sharpe_proxy_21 | 2.7077 | -0.0418 | 1.267 | -0.209 |
| expectancy_proxy_21 | 0.1576 | 0.0042 | 1.226 | -0.262 |
| regime_risk | 0.4198 | 0.5998 | -1.167 | 0.205 |
| trend_slope_63 | 0.1377 | -0.0033 | 1.162 | 0.014 |
| cum_drop_30 | -0.0856 | -0.2304 | 1.110 | -0.286 |
| realized_vol_63 | 0.8339 | 1.0763 | -1.096 | 0.031 |
| rsi_14 | 0.5975 | 0.4744 | 1.072 | -0.197 |
| below_anchor_ratio_63 | 0.0594 | 0.4676 | -0.974 | 0.026 |
| mom_21 | 0.1761 | -0.0236 | 0.970 | -0.209 |
| relative_strength_21 | 1.1362 | -0.1971 | 0.967 | -0.205 |
| downside_semivol_21 | 0.5032 | 0.7707 | -0.951 | 0.132 |
| vix_zscore | -0.7058 | 0.5322 | -0.936 | 0.226 |
| drawdown_21 | -0.0773 | -0.1915 | 0.921 | -0.329 |
| trend_slope_21 | 0.0650 | -0.0217 | 0.920 | 0.002 |
| cum_drop_20 | -0.0759 | -0.1868 | 0.899 | -0.324 |
| trend_gap_21 | 0.0579 | -0.0301 | 0.864 | 0.007 |
| vix_level | 0.2737 | 0.3342 | -0.821 | 0.210 |
| trend_gap_63 | 0.1149 | -0.0170 | 0.820 | 0.017 |
| anchor_gap_velocity_10 | 0.0681 | -0.0857 | 0.745 | -0.240 |
| drawdown_streak_63 | 0.3361 | 0.6090 | -0.739 | 0.031 |
| below_ma63_days | 0.0388 | 0.2195 | -0.713 | 0.037 |
| days_since_high_63 | 0.3076 | 0.5271 | -0.711 | 0.016 |
| mom_10 | 0.0971 | -0.0265 | 0.707 | -0.254 |
| realized_vol_21 | 0.8340 | 1.0651 | -0.701 | 0.049 |
| vol_stress_score | 0.7088 | 0.4958 | 0.701 | 0.091 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
