# OOS Poor-Window Diagnostic: base_multi_obj_plain__wf_2023_2024

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
| 2024-04-19 | 2024-06-18 | 27.46% | 111.89% | -84.43pp | 2.92% | 0.308 | 114.43 |
| 2024-04-22 | 2024-06-20 | 22.23% | 87.15% | -64.92pp | 3.60% | 0.315 | 94.92 |
| 2024-04-18 | 2024-06-17 | 23.37% | 79.52% | -56.15pp | 2.92% | 0.300 | 86.15 |
| 2024-01-05 | 2024-03-06 | 39.77% | 93.68% | -53.91pp | 4.35% | 0.439 | 83.91 |
| 2024-01-08 | 2024-03-07 | 42.15% | 94.58% | -52.44pp | 4.35% | 0.443 | 82.44 |
| 2024-01-03 | 2024-03-04 | 38.59% | 89.60% | -51.01pp | 4.35% | 0.432 | 81.01 |
| 2024-04-23 | 2024-06-21 | 19.65% | 70.62% | -50.98pp | 4.85% | 0.322 | 80.98 |
| 2024-01-04 | 2024-03-05 | 35.78% | 83.11% | -47.33pp | 4.35% | 0.435 | 77.33 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| negative_gap_frequency_21 | 0.2466 | 0.3623 | -1.393 | 0.082 |
| vix_percentile_252 | 0.2708 | 0.6483 | -1.336 | 0.192 |
| drawdown_63 | -0.1182 | -0.3141 | 1.325 | -0.148 |
| expectancy_proxy_21 | 0.1598 | 0.0049 | 1.245 | -0.235 |
| regime_risk | 0.4129 | 0.6014 | -1.236 | 0.172 |
| sharpe_proxy_21 | 2.6902 | 0.0021 | 1.226 | -0.181 |
| anchor_gap_180 | 0.4988 | 0.0364 | 1.219 | -0.136 |
| mom_63 | 0.4621 | 0.0946 | 1.161 | -0.055 |
| cum_drop_30 | -0.0846 | -0.2292 | 1.112 | -0.257 |
| realized_vol_63 | 0.8314 | 1.0747 | -1.105 | -0.001 |
| trend_slope_63 | 0.1334 | 0.0008 | 1.060 | 0.036 |
| rsi_14 | 0.5966 | 0.4764 | 1.041 | -0.181 |
| downside_semivol_21 | 0.5000 | 0.7692 | -0.963 | 0.107 |
| below_anchor_ratio_63 | 0.0607 | 0.4618 | -0.956 | 0.006 |
| vix_zscore | -0.7160 | 0.5226 | -0.940 | 0.196 |
| drawdown_21 | -0.0761 | -0.1908 | 0.929 | -0.302 |
| mom_21 | 0.1721 | -0.0189 | 0.915 | -0.184 |
| relative_strength_21 | 1.1097 | -0.1658 | 0.913 | -0.180 |
| cum_drop_20 | -0.0746 | -0.1861 | 0.909 | -0.295 |
| trend_slope_21 | 0.0626 | -0.0193 | 0.851 | 0.020 |
| vix_level | 0.2730 | 0.3338 | -0.831 | 0.179 |
| trend_gap_21 | 0.0549 | -0.0274 | 0.790 | 0.024 |
| trend_gap_63 | 0.1120 | -0.0138 | 0.775 | 0.037 |
| realized_vol_21 | 0.8253 | 1.0671 | -0.740 | 0.021 |
| below_ma63_days | 0.0397 | 0.2168 | -0.700 | 0.013 |
| rsi_6 | 0.6196 | 0.4760 | 0.699 | -0.599 |
| drawdown_streak_63 | 0.3429 | 0.6018 | -0.695 | 0.008 |
| anchor_gap_velocity_10 | 0.0633 | -0.0811 | 0.691 | -0.231 |
| anchor_slope_180 | 0.0850 | 0.0447 | 0.668 | 0.043 |
| days_since_high_63 | 0.3138 | 0.5210 | -0.665 | -0.002 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
