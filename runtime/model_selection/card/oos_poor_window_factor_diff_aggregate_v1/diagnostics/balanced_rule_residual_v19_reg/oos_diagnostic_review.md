# OOS Poor-Window Diagnostic: balanced_rule_residual_v19_reg

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
| 2025-04-08 | 2025-07-09 | 42.55% | 223.03% | -180.48pp | 6.70% | 0.556 | 187.18 |
| 2025-04-21 | 2025-07-21 | 43.99% | 215.73% | -171.73pp | 4.59% | 0.564 | 176.32 |
| 2020-03-20 | 2020-06-18 | 20.45% | 188.85% | -168.40pp | 6.58% | 0.573 | 174.97 |
| 2020-03-18 | 2020-06-16 | 19.36% | 177.82% | -158.46pp | 6.58% | 0.562 | 165.04 |
| 2025-04-17 | 2025-07-18 | 42.25% | 197.07% | -154.82pp | 4.59% | 0.565 | 159.41 |
| 2025-04-04 | 2025-07-07 | 36.11% | 186.03% | -149.92pp | 6.70% | 0.554 | 156.62 |
| 2025-04-07 | 2025-07-08 | 39.44% | 188.20% | -148.76pp | 6.70% | 0.555 | 155.46 |
| 2025-04-16 | 2025-07-17 | 41.87% | 192.51% | -150.63pp | 4.59% | 0.564 | 155.23 |
| 2025-04-22 | 2025-07-22 | 39.27% | 186.71% | -147.44pp | 4.59% | 0.563 | 152.03 |
| 2020-03-19 | 2020-06-17 | 20.68% | 165.38% | -144.70pp | 6.58% | 0.568 | 151.27 |
| 2025-04-10 | 2025-07-11 | 41.09% | 183.39% | -142.29pp | 4.59% | 0.559 | 146.88 |
| 2020-03-23 | 2020-06-19 | 19.47% | 158.94% | -139.47pp | 6.58% | 0.576 | 146.05 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| vol_of_vol_21 | 0.3692 | 0.1031 | 4.523 | -0.027 |
| tracking_error_vol_21 | 0.1873 | 0.0579 | 4.457 | 0.006 |
| decay_63 | -0.8447 | -0.1793 | -4.429 | 0.069 |
| realized_vol_63 | 1.8331 | 0.9725 | 3.670 | -0.016 |
| tracking_error_21 | -0.0258 | -0.0059 | -2.768 | 0.092 |
| vol_coef_21 | 1.8625 | 1.0573 | 2.263 | -0.003 |
| realized_vol_21 | 1.6415 | 0.9717 | 2.242 | -0.002 |
| vix_level | 0.6040 | 0.4011 | 1.617 | 0.096 |
| trend_slope_63 | -0.1536 | 0.0513 | -1.547 | 0.014 |
| jump_variation_5 | 1.3262 | 0.1071 | 1.528 | 0.008 |
| realized_vol_10 | 1.5219 | 0.9656 | 1.488 | -0.004 |
| rolling_new_low_ratio_63 | 0.1129 | 0.0374 | 1.327 | -0.013 |
| downside_semivol_21 | 0.9971 | 0.6505 | 1.296 | 0.061 |
| below_anchor_ratio_63 | 0.8272 | 0.3357 | 1.210 | 0.004 |
| drawdown_63 | -0.4012 | -0.2032 | -1.094 | -0.181 |
| anchor_gap_180 | -0.3082 | 0.1953 | -1.065 | -0.101 |
| anchor_slope_180 | -0.0548 | 0.0322 | -0.973 | -0.002 |
| vix_percentile_252 | 0.7469 | 0.4486 | 0.942 | 0.112 |
| realized_skew_21 | 0.4381 | -0.1154 | 0.907 | 0.007 |
| trend_gap_63 | -0.1038 | 0.0407 | -0.898 | 0.009 |
| vix_zscore | 1.0095 | -0.1162 | 0.883 | 0.104 |
| tracking_error_5 | -0.0232 | -0.0062 | -0.838 | 0.136 |
| realized_kurtosis_21 | 1.3942 | 0.2873 | 0.838 | -0.020 |
| regime_risk | 0.5798 | 0.4486 | 0.667 | 0.121 |
| below_ma63_days | 0.3117 | 0.1435 | 0.619 | 0.072 |
| days_since_high_63 | 0.5769 | 0.3719 | 0.606 | 0.030 |
| mom_63 | 0.0363 | 0.2388 | -0.511 | -0.093 |
| intraday_range | 0.0939 | 0.0748 | 0.480 | 0.168 |
| regime_risk_no_vix | 0.5393 | 0.4637 | 0.453 | 0.087 |
| drawdown_streak_63 | 0.6159 | 0.4408 | 0.432 | 0.039 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
