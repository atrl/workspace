# OOS Poor-Window Diagnostic: drawdown_rebound_reward_train_v1:low_dd_iter0

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
| 2025-04-08 | 2025-07-09 | 39.80% | 223.03% | -183.23pp | 11.31% | 0.248 | 194.54 |
| 2025-04-21 | 2025-07-21 | 38.32% | 215.73% | -177.41pp | 4.56% | 0.287 | 181.98 |
| 2020-03-20 | 2020-06-18 | 24.04% | 188.85% | -164.81pp | 8.42% | 0.418 | 173.22 |
| 2025-04-07 | 2025-07-08 | 28.18% | 188.20% | -160.02pp | 11.31% | 0.243 | 171.33 |
| 2025-04-04 | 2025-07-07 | 27.61% | 186.03% | -158.42pp | 11.31% | 0.238 | 169.73 |
| 2025-04-17 | 2025-07-18 | 36.32% | 197.07% | -160.75pp | 4.56% | 0.282 | 165.31 |
| 2020-03-18 | 2020-06-16 | 22.96% | 177.82% | -154.85pp | 8.42% | 0.414 | 163.27 |
| 2025-04-16 | 2025-07-17 | 35.80% | 192.51% | -156.71pp | 4.56% | 0.277 | 161.27 |
| 2026-02-05 | 2026-05-06 | 75.97% | 211.46% | -135.48pp | 23.16% | 0.096 | 158.64 |
| 2025-04-22 | 2025-07-22 | 35.35% | 186.71% | -151.36pp | 4.56% | 0.292 | 155.93 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| decay_63 | -0.6611 | -0.1763 | -3.171 | 0.066 |
| vol_of_vol_21 | 0.2819 | 0.1040 | 2.993 | -0.025 |
| tracking_error_vol_21 | 0.1399 | 0.0590 | 2.775 | 0.006 |
| realized_vol_63 | 1.5989 | 0.9682 | 2.644 | -0.013 |
| tracking_error_21 | -0.0204 | -0.0058 | -2.000 | 0.088 |
| realized_vol_21 | 1.4899 | 0.9639 | 1.741 | 0.003 |
| vol_coef_21 | 1.6358 | 1.0543 | 1.603 | 0.002 |
| vix_level | 0.5500 | 0.3999 | 1.179 | 0.098 |
| realized_vol_10 | 1.3971 | 0.9589 | 1.170 | -0.000 |
| downside_semivol_21 | 0.9229 | 0.6459 | 1.024 | 0.070 |
| jump_variation_5 | 0.9192 | 0.1118 | 0.991 | 0.007 |
| vix_percentile_252 | 0.7470 | 0.4354 | 0.987 | 0.118 |
| trend_slope_63 | -0.0669 | 0.0479 | -0.857 | 0.010 |
| vix_zscore | 0.8440 | -0.1421 | 0.765 | 0.109 |
| rolling_new_low_ratio_63 | 0.0804 | 0.0387 | 0.721 | -0.009 |
| realized_kurtosis_21 | 1.1300 | 0.2764 | 0.635 | -0.017 |
| regime_risk | 0.5690 | 0.4444 | 0.628 | 0.130 |
| drawdown_63 | -0.3202 | -0.2061 | -0.625 | -0.195 |
| tracking_error_5 | -0.0187 | -0.0061 | -0.609 | 0.138 |
| realized_skew_21 | 0.2301 | -0.1099 | 0.546 | 0.004 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
