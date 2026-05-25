# OOS Poor-Window Diagnostic: drawdown_rebound_reward_train_v1:summary_selected_strict

## Scope

- train_end: `2018-12-31`
- OOS window: `2020-01-02` .. `2026-05-21`
- target_excess=30.00%
- target_max_dd=15.00%

## External References

- [Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf): selection-bias warning for repeated trials
- [Probability of Backtest Overfitting](https://doi.org/10.21314/JCF.2016.322): strategy-selection overfit diagnostic
- [Qlib GitHub](https://github.com/microsoft/qlib): repo-owned qlib workflow

## Poor Windows

| start | end | strategy | buy-hold | excess | max_dd | avg_pos | bad_score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-04-08 | 2025-07-09 | 56.67% | 223.03% | -166.36pp | 12.40% | 0.568 | 178.75 |
| 2020-03-20 | 2020-06-18 | 22.45% | 188.85% | -166.41pp | 10.97% | 0.624 | 177.37 |
| 2020-03-18 | 2020-06-16 | 21.03% | 177.82% | -156.79pp | 10.97% | 0.626 | 167.76 |
| 2025-04-21 | 2025-07-21 | 56.54% | 215.73% | -159.19pp | 5.44% | 0.587 | 164.63 |
| 2025-04-04 | 2025-07-07 | 37.86% | 186.03% | -148.16pp | 12.40% | 0.562 | 160.56 |
| 2025-04-07 | 2025-07-08 | 41.30% | 188.20% | -146.90pp | 12.40% | 0.565 | 159.30 |
| 2020-03-19 | 2020-06-17 | 22.74% | 165.38% | -142.64pp | 10.97% | 0.625 | 153.61 |
| 2025-04-17 | 2025-07-18 | 53.71% | 197.07% | -143.35pp | 5.44% | 0.586 | 148.79 |
| 2020-03-23 | 2020-06-19 | 21.17% | 158.94% | -137.77pp | 10.97% | 0.623 | 148.74 |
| 2025-04-16 | 2025-07-17 | 53.05% | 192.51% | -139.46pp | 5.44% | 0.584 | 144.90 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| vol_of_vol_21 | 0.3712 | 0.1031 | 4.559 | -0.027 |
| tracking_error_vol_21 | 0.1884 | 0.0579 | 4.495 | 0.001 |
| decay_63 | -0.8489 | -0.1793 | -4.458 | 0.068 |
| realized_vol_63 | 1.8402 | 0.9724 | 3.701 | -0.020 |
| tracking_error_21 | -0.0259 | -0.0059 | -2.782 | 0.089 |
| vol_coef_21 | 1.8719 | 1.0569 | 2.290 | -0.007 |
| realized_vol_21 | 1.6493 | 0.9714 | 2.268 | -0.005 |
| vix_level | 0.6060 | 0.4010 | 1.634 | 0.091 |
| trend_slope_63 | -0.1566 | 0.0514 | -1.570 | 0.017 |
| jump_variation_5 | 1.3358 | 0.1070 | 1.541 | 0.005 |
| realized_vol_10 | 1.5303 | 0.9651 | 1.511 | -0.009 |
| rolling_new_low_ratio_63 | 0.1137 | 0.0374 | 1.342 | -0.016 |
| downside_semivol_21 | 1.0023 | 0.6503 | 1.316 | 0.052 |
| below_anchor_ratio_63 | 0.8281 | 0.3359 | 1.211 | 0.003 |
| drawdown_63 | -0.4038 | -0.2031 | -1.109 | -0.166 |
| anchor_gap_180 | -0.3113 | 0.1953 | -1.072 | -0.093 |
| anchor_slope_180 | -0.0549 | 0.0321 | -0.973 | 0.000 |
| vix_percentile_252 | 0.7500 | 0.4485 | 0.953 | 0.107 |
| trend_gap_63 | -0.1068 | 0.0409 | -0.918 | 0.018 |
| realized_skew_21 | 0.4358 | -0.1148 | 0.902 | 0.013 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
