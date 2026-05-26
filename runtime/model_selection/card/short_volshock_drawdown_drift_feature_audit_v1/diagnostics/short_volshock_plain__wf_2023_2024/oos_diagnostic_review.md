# OOS Poor-Window Diagnostic: short_volshock_plain__wf_2023_2024

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
| 2024-04-19 | 2024-06-18 | 12.37% | 111.89% | -99.52pp | 2.83% | 0.173 | 129.52 |
| 2024-04-22 | 2024-06-20 | 8.58% | 87.15% | -78.57pp | 3.12% | 0.174 | 108.57 |
| 2024-01-05 | 2024-03-06 | 22.71% | 93.68% | -70.96pp | 5.92% | 0.354 | 100.96 |
| 2024-01-08 | 2024-03-07 | 24.34% | 94.58% | -70.24pp | 5.92% | 0.357 | 100.24 |
| 2024-04-18 | 2024-06-17 | 9.42% | 79.52% | -70.10pp | 2.83% | 0.169 | 100.10 |
| 2024-01-03 | 2024-03-04 | 21.56% | 89.60% | -68.04pp | 5.92% | 0.349 | 98.04 |
| 2024-01-04 | 2024-03-05 | 19.14% | 83.11% | -63.97pp | 5.92% | 0.351 | 93.97 |
| 2024-04-23 | 2024-06-21 | 7.83% | 70.62% | -62.79pp | 3.41% | 0.175 | 92.79 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| negative_gap_frequency_21 | 0.2466 | 0.3623 | -1.393 | 0.095 |
| vix_percentile_252 | 0.2708 | 0.6483 | -1.336 | 0.216 |
| drawdown_63 | -0.1182 | -0.3141 | 1.325 | -0.166 |
| expectancy_proxy_21 | 0.1598 | 0.0049 | 1.245 | -0.255 |
| regime_risk | 0.4129 | 0.6014 | -1.236 | 0.192 |
| sharpe_proxy_21 | 2.6902 | 0.0021 | 1.226 | -0.202 |
| anchor_gap_180 | 0.4988 | 0.0364 | 1.219 | -0.152 |
| mom_63 | 0.4621 | 0.0946 | 1.161 | -0.063 |
| cum_drop_30 | -0.0846 | -0.2292 | 1.112 | -0.274 |
| realized_vol_63 | 0.8314 | 1.0747 | -1.105 | 0.012 |
| trend_slope_63 | 0.1334 | 0.0008 | 1.060 | 0.034 |
| rsi_14 | 0.5966 | 0.4764 | 1.041 | -0.191 |
| downside_semivol_21 | 0.5000 | 0.7692 | -0.963 | 0.121 |
| below_anchor_ratio_63 | 0.0607 | 0.4618 | -0.956 | 0.020 |
| vix_zscore | -0.7160 | 0.5226 | -0.940 | 0.220 |
| drawdown_21 | -0.0761 | -0.1908 | 0.929 | -0.317 |
| mom_21 | 0.1721 | -0.0189 | 0.915 | -0.202 |
| relative_strength_21 | 1.1097 | -0.1658 | 0.913 | -0.198 |
| cum_drop_20 | -0.0746 | -0.1861 | 0.909 | -0.312 |
| trend_slope_21 | 0.0626 | -0.0193 | 0.851 | 0.012 |
| vix_level | 0.2730 | 0.3338 | -0.831 | 0.205 |
| trend_gap_21 | 0.0549 | -0.0274 | 0.790 | 0.013 |
| trend_gap_63 | 0.1120 | -0.0138 | 0.775 | 0.032 |
| realized_vol_21 | 0.8253 | 1.0671 | -0.740 | 0.034 |
| below_ma63_days | 0.0397 | 0.2168 | -0.700 | 0.026 |
| rsi_6 | 0.6196 | 0.4760 | 0.699 | -0.615 |
| drawdown_streak_63 | 0.3429 | 0.6018 | -0.695 | 0.015 |
| anchor_gap_velocity_10 | 0.0633 | -0.0811 | 0.691 | -0.235 |
| anchor_slope_180 | 0.0850 | 0.0447 | 0.668 | 0.029 |
| days_since_high_63 | 0.3138 | 0.5210 | -0.665 | 0.004 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
