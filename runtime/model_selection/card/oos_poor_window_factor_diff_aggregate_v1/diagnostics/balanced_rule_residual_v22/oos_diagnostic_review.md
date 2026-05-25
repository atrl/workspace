# OOS Poor-Window Diagnostic: balanced_rule_residual_v22

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
| 2020-03-20 | 2020-06-18 | 20.42% | 188.85% | -168.43pp | 13.89% | 0.604 | 182.31 |
| 2025-04-08 | 2025-07-09 | 51.74% | 223.03% | -171.29pp | 8.12% | 0.610 | 179.41 |
| 2020-03-18 | 2020-06-16 | 18.49% | 177.82% | -159.33pp | 13.89% | 0.614 | 173.21 |
| 2025-04-21 | 2025-07-21 | 53.29% | 215.73% | -162.44pp | 5.46% | 0.611 | 167.90 |
| 2020-03-19 | 2020-06-17 | 20.83% | 165.38% | -144.55pp | 13.89% | 0.609 | 158.44 |
| 2020-03-23 | 2020-06-19 | 19.72% | 158.94% | -139.22pp | 13.89% | 0.600 | 153.11 |
| 2025-04-17 | 2025-07-18 | 51.12% | 197.07% | -145.94pp | 5.46% | 0.610 | 151.40 |
| 2025-04-04 | 2025-07-07 | 43.65% | 186.03% | -142.38pp | 8.12% | 0.610 | 150.50 |
| 2025-04-07 | 2025-07-08 | 47.72% | 188.20% | -140.48pp | 8.12% | 0.610 | 148.59 |
| 2025-04-16 | 2025-07-17 | 50.64% | 192.51% | -141.87pp | 5.46% | 0.609 | 147.33 |
| 2026-02-05 | 2026-05-06 | 89.02% | 211.46% | -122.43pp | 24.12% | 0.132 | 146.55 |
| 2025-04-22 | 2025-07-22 | 47.94% | 186.71% | -138.77pp | 5.46% | 0.612 | 144.22 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| decay_63 | -0.6579 | -0.1764 | -3.150 | 0.067 |
| vol_of_vol_21 | 0.2810 | 0.1040 | 2.977 | -0.031 |
| tracking_error_vol_21 | 0.1396 | 0.0590 | 2.762 | 0.004 |
| realized_vol_63 | 1.5988 | 0.9678 | 2.651 | -0.017 |
| tracking_error_21 | -0.0203 | -0.0058 | -1.986 | 0.090 |
| realized_vol_21 | 1.4875 | 0.9639 | 1.732 | -0.004 |
| vol_coef_21 | 1.6335 | 1.0542 | 1.597 | -0.002 |
| vix_level | 0.5507 | 0.3997 | 1.188 | 0.094 |
| realized_vol_10 | 1.3957 | 0.9588 | 1.166 | -0.005 |
| downside_semivol_21 | 0.9220 | 0.6458 | 1.021 | 0.057 |
| vix_percentile_252 | 0.7474 | 0.4351 | 0.990 | 0.112 |
| jump_variation_5 | 0.9146 | 0.1119 | 0.985 | 0.002 |
| trend_slope_63 | -0.0666 | 0.0479 | -0.855 | 0.017 |
| vix_zscore | 0.8438 | -0.1428 | 0.765 | 0.104 |
| rolling_new_low_ratio_63 | 0.0800 | 0.0388 | 0.714 | -0.014 |
| realized_kurtosis_21 | 1.1458 | 0.2735 | 0.651 | -0.022 |
| regime_risk | 0.5689 | 0.4443 | 0.628 | 0.119 |
| drawdown_63 | -0.3190 | -0.2062 | -0.617 | -0.176 |
| tracking_error_5 | -0.0185 | -0.0061 | -0.600 | 0.134 |
| below_anchor_ratio_63 | 0.5726 | 0.3505 | 0.543 | 0.002 |
| realized_skew_21 | 0.2202 | -0.1087 | 0.530 | 0.012 |
| mom_21 | 0.1961 | 0.0635 | 0.524 | -0.176 |
| mom_10 | 0.1201 | 0.0269 | 0.511 | -0.260 |
| relative_strength_21 | 1.2363 | 0.4028 | 0.493 | -0.174 |
| expectancy_proxy_21 | 0.1614 | 0.0694 | 0.485 | -0.197 |
| downside_ratio_10 | 0.4084 | 0.4732 | -0.458 | 0.239 |
| trend_gap_63 | -0.0361 | 0.0374 | -0.457 | 0.014 |
| downside_ratio_21 | 0.4302 | 0.4697 | -0.451 | 0.182 |
| mom_5 | 0.0709 | 0.0117 | 0.439 | -0.363 |
| sharpe_proxy_21 | 2.4986 | 1.1083 | 0.439 | -0.173 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
