# OOS Poor-Window Diagnostic: drawdown_rebound_reward_train_v1

## Scope

- train_end: `2018-12-31`
- OOS window: `2020-01-02` .. `2026-05-21`
- target_excess=30.00%
- target_max_dd=15.00%

## External References

- [Qlib arXiv](https://arxiv.org/abs/2009.11189): AI quant workflow with data, learning, backtest, and deployment surfaces
- [Qlib GitHub](https://github.com/microsoft/qlib): repo-compatible qlib research-to-production workflow
- [Probability of Backtest Overfitting](https://scholarworks.wmich.edu/math_pubs/42/): use OOS and cross-window evidence before selection
- [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551): multiple-testing and selection-bias discipline
- [MetaTrader](https://arxiv.org/abs/2210.01774): market-condition-specific policy integration
- [Behavior Cloning Trading Agents](https://arxiv.org/abs/2110.01325): behavior cloning as a supervised policy-refresh mechanism

## Poor Windows

| start | end | strategy | buy-hold | excess | max_dd | avg_pos | bad_score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-04-08 | 2025-07-09 | 23.64% | 223.03% | -199.39pp | 4.03% | 0.421 | 203.42 |
| 2025-04-21 | 2025-07-21 | 26.85% | 215.73% | -188.88pp | 4.03% | 0.422 | 192.91 |
| 2026-02-05 | 2026-05-06 | 40.35% | 211.46% | -171.11pp | 13.26% | 0.059 | 184.37 |
| 2020-03-20 | 2020-06-18 | 14.06% | 188.85% | -174.79pp | 5.27% | 0.353 | 180.06 |
| 2026-02-10 | 2026-05-11 | 35.18% | 200.63% | -165.45pp | 13.26% | 0.040 | 178.71 |
| 2025-04-17 | 2025-07-18 | 23.64% | 197.07% | -173.42pp | 4.03% | 0.420 | 177.45 |
| 2025-04-16 | 2025-07-17 | 23.40% | 192.51% | -169.11pp | 4.03% | 0.420 | 173.13 |
| 2026-02-12 | 2026-05-13 | 33.97% | 191.98% | -158.01pp | 13.26% | 0.031 | 171.28 |
| 2020-03-18 | 2020-06-16 | 12.85% | 177.82% | -164.96pp | 5.27% | 0.350 | 170.23 |
| 2025-04-07 | 2025-07-08 | 22.45% | 188.20% | -165.75pp | 4.03% | 0.421 | 169.77 |
| 2025-04-04 | 2025-07-07 | 21.03% | 186.03% | -165.00pp | 4.03% | 0.421 | 169.02 |
| 2025-04-22 | 2025-07-22 | 22.33% | 186.71% | -164.38pp | 4.03% | 0.426 | 168.41 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| decay_63 | -0.6511 | -0.1761 | -3.103 | 0.064 |
| vol_of_vol_21 | 0.2781 | 0.1039 | 2.924 | -0.033 |
| tracking_error_vol_21 | 0.1378 | 0.0590 | 2.696 | 0.011 |
| realized_vol_63 | 1.5896 | 0.9673 | 2.609 | -0.015 |
| tracking_error_21 | -0.0201 | -0.0058 | -1.947 | 0.090 |
| realized_vol_21 | 1.4819 | 0.9632 | 1.715 | -0.003 |
| vol_coef_21 | 1.6238 | 1.0540 | 1.569 | -0.004 |
| realized_vol_10 | 1.3995 | 0.9570 | 1.183 | -0.002 |
| vix_level | 0.5452 | 0.4000 | 1.138 | 0.101 |
| downside_semivol_21 | 0.9137 | 0.6462 | 0.987 | 0.064 |
| vix_percentile_252 | 0.7425 | 0.4350 | 0.973 | 0.121 |
| jump_variation_5 | 0.8970 | 0.1122 | 0.961 | 0.007 |
| trend_slope_63 | -0.0572 | 0.0468 | -0.782 | 0.011 |
| vix_zscore | 0.8196 | -0.1420 | 0.744 | 0.112 |
| rolling_new_low_ratio_63 | 0.0788 | 0.0388 | 0.692 | -0.010 |
| regime_risk | 0.5673 | 0.4442 | 0.619 | 0.127 |
| mom_21 | 0.2131 | 0.0606 | 0.613 | -0.186 |
| realized_kurtosis_21 | 1.1027 | 0.2774 | 0.613 | -0.019 |
| tracking_error_5 | -0.0184 | -0.0061 | -0.596 | 0.135 |
| drawdown_63 | -0.3135 | -0.2067 | -0.585 | -0.195 |
| relative_strength_21 | 1.3501 | 0.3836 | 0.583 | -0.184 |
| mom_10 | 0.1286 | 0.0254 | 0.570 | -0.274 |
| expectancy_proxy_21 | 0.1713 | 0.0676 | 0.553 | -0.203 |
| downside_ratio_21 | 0.4254 | 0.4705 | -0.523 | 0.179 |
| realized_skew_21 | 0.2133 | -0.1086 | 0.517 | 0.012 |
| sharpe_proxy_21 | 2.6644 | 1.0798 | 0.506 | -0.182 |
| below_anchor_ratio_63 | 0.5571 | 0.3521 | 0.500 | 0.011 |
| mom_5 | 0.0754 | 0.0108 | 0.482 | -0.376 |
| downside_ratio_10 | 0.4068 | 0.4736 | -0.473 | 0.242 |
| anchor_gap_velocity_10 | 0.0913 | -0.0034 | 0.464 | -0.254 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
