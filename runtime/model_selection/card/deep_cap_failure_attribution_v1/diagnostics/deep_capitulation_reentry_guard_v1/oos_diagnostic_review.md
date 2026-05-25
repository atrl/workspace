# OOS Poor-Window Diagnostic: deep_capitulation_reentry_guard_v1

## Scope

- train_end: `2018-12-31`
- OOS window: `2020-01-02` .. `2026-05-21`
- target_excess=30.00%
- target_max_dd=15.00%

## External References

- [Qlib: An AI-oriented Quantitative Investment Platform](https://arxiv.org/abs/2009.11189): research-to-production workflow and persisted quant artifacts
- [Probability of Backtest Overfitting](https://doi.org/10.21314/JCF.2016.322): use OOS/walk-forward evidence before promoting repeated trials
- [Focal Loss](https://arxiv.org/abs/1708.02002): imbalance-aware learning reference already tested in prior rebound-loss screens
- [Class-Balanced Loss](https://arxiv.org/abs/1901.05555): effective-number reweighting reference; do not repeat if failure is not weight-only

## Poor Windows

| start | end | strategy | buy-hold | excess | max_dd | avg_pos | bad_score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-02-10 | 2026-05-11 | 3.69% | 200.63% | -196.94pp | 5.90% | 0.122 | 202.84 |
| 2026-02-05 | 2026-05-06 | 19.57% | 211.46% | -191.89pp | 5.90% | 0.161 | 197.79 |
| 2026-02-12 | 2026-05-13 | 3.92% | 191.98% | -188.06pp | 4.54% | 0.096 | 192.60 |
| 2026-02-13 | 2026-05-14 | 1.97% | 188.26% | -186.30pp | 4.54% | 0.083 | 190.83 |
| 2026-02-09 | 2026-05-08 | 2.78% | 176.34% | -173.56pp | 5.90% | 0.135 | 179.46 |
| 2026-02-23 | 2026-05-21 | -0.24% | 170.86% | -171.10pp | 4.54% | 0.024 | 175.63 |
| 2025-04-08 | 2025-07-09 | 70.23% | 223.03% | -152.80pp | 12.17% | 0.505 | 164.98 |
| 2026-02-20 | 2026-05-20 | -1.16% | 158.08% | -159.25pp | 4.54% | 0.032 | 163.78 |
| 2026-02-11 | 2026-05-12 | -2.21% | 153.52% | -155.73pp | 5.90% | 0.109 | 161.63 |
| 2025-04-21 | 2025-07-21 | 67.39% | 215.73% | -148.33pp | 12.17% | 0.571 | 160.51 |
| 2026-02-17 | 2026-05-15 | 2.28% | 155.14% | -152.86pp | 4.54% | 0.070 | 157.40 |
| 2026-02-04 | 2026-05-05 | 19.68% | 171.03% | -151.35pp | 5.90% | 0.173 | 157.25 |
| 2026-02-06 | 2026-05-07 | 5.86% | 146.32% | -140.46pp | 5.90% | 0.148 | 146.36 |
| 2025-04-17 | 2025-07-18 | 64.37% | 197.07% | -132.70pp | 12.17% | 0.565 | 144.87 |
| 2025-04-16 | 2025-07-17 | 63.71% | 192.51% | -128.80pp | 12.17% | 0.556 | 140.97 |
| 2025-04-22 | 2025-07-22 | 60.75% | 186.71% | -125.96pp | 12.17% | 0.577 | 138.13 |
| 2025-04-04 | 2025-07-07 | 60.78% | 186.03% | -125.24pp | 12.17% | 0.489 | 137.42 |
| 2026-02-19 | 2026-05-19 | 1.15% | 132.85% | -131.71pp | 4.54% | 0.045 | 136.24 |
| 2025-04-07 | 2025-07-08 | 64.87% | 188.20% | -123.33pp | 12.17% | 0.496 | 135.51 |
| 2026-02-18 | 2026-05-18 | -0.17% | 128.92% | -129.08pp | 4.54% | 0.058 | 133.62 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| jump_variation_5 | 1.2461 | 0.1078 | 1.423 | 0.006 |
| vol_of_vol_21 | 0.2241 | 0.1164 | 1.168 | -0.022 |
| realized_vol_63 | 1.3671 | 1.0149 | 1.079 | -0.009 |
| realized_kurtosis_21 | 1.5315 | 0.2664 | 0.947 | -0.040 |
| downside_ratio_21 | 0.4052 | 0.4708 | -0.767 | 0.191 |
| mom_21 | 0.2603 | 0.0618 | 0.763 | -0.189 |
| expectancy_proxy_21 | 0.2090 | 0.0679 | 0.758 | -0.212 |
| relative_strength_21 | 1.6995 | 0.3864 | 0.752 | -0.187 |
| trend_slope_21 | 0.0987 | 0.0139 | 0.739 | -0.010 |
| vix_percentile_252 | 0.6877 | 0.4528 | 0.730 | 0.113 |
| realized_vol_21 | 1.2799 | 1.0046 | 0.701 | 0.001 |
| sharpe_proxy_21 | 3.2135 | 1.0867 | 0.685 | -0.189 |
| realized_skew_21 | 0.3053 | -0.1052 | 0.663 | 0.007 |
| trend_gap_21 | 0.0785 | 0.0076 | 0.595 | -0.032 |
| mom_10 | 0.1361 | 0.0287 | 0.574 | -0.269 |
| anchor_gap_velocity_10 | 0.1139 | -0.0020 | 0.571 | -0.250 |
| downside_ratio_10 | 0.3946 | 0.4722 | -0.552 | 0.250 |
| gain_streak_10 | 0.1832 | 0.1059 | 0.532 | -0.783 |
| realized_vol_10 | 1.2280 | 0.9922 | 0.531 | 0.007 |
| rsi_14 | 0.6070 | 0.5266 | 0.523 | -0.200 |
| negative_gap_frequency_21 | 0.4017 | 0.3434 | 0.485 | 0.093 |
| vix_zscore | 0.5270 | -0.0737 | 0.451 | 0.104 |
| mom_63 | 0.3799 | 0.2049 | 0.435 | -0.098 |
| rolling_new_low_ratio_63 | 0.0666 | 0.0417 | 0.428 | -0.006 |
| mom_5 | 0.0697 | 0.0140 | 0.404 | -0.364 |
| anchor_gap_180 | 0.3044 | 0.1357 | 0.386 | -0.115 |
| decay_63 | -0.3367 | -0.2272 | -0.381 | 0.064 |
| rsi_6 | 0.6178 | 0.5313 | 0.377 | -0.577 |
| trend_gap_63 | 0.0793 | 0.0228 | 0.335 | 0.010 |
| vol_coef_21 | 1.2731 | 1.1126 | 0.319 | -0.001 |
| new_low_ratio_21 | 0.0578 | 0.0916 | -0.285 | 0.024 |
| below_anchor_ratio_63 | 0.4805 | 0.3681 | 0.274 | 0.013 |
| vol_stress_score | 0.5937 | 0.5135 | 0.273 | -0.020 |
| regime_risk | 0.5086 | 0.4551 | 0.263 | 0.126 |
| intraday_range | 0.0861 | 0.0755 | 0.258 | 0.174 |
| downside_semivol_21 | 0.7469 | 0.6740 | 0.222 | 0.069 |
| drawdown_streak_63 | 0.3741 | 0.4644 | -0.222 | 0.055 |
| return | 0.0163 | 0.0026 | 0.201 | -0.971 |
| tracking_error_21 | -0.0102 | -0.0074 | -0.179 | 0.090 |
| loss_streak_21 | 0.0313 | 0.0404 | -0.163 | 0.824 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
