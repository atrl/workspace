# OOS Poor-Window Diagnostic: recovery_reentry_2021_trendbreak_cap_v1

## Scope

- train_end: `2018-12-31`
- OOS window: `2020-01-02` .. `2026-05-21`
- target_excess=30.00%
- target_max_dd=15.00%

## External References

- [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551): selection-bias and multiple-testing warning for repeated strategy trials
- [Probability of Backtest Overfitting](https://doi.org/10.21314/JCF.2016.322): walk-forward and OOS discipline motivation
- [Qlib GitHub](https://github.com/microsoft/qlib): keep diagnostics inside the qlib research-to-production workflow
- [Qlib arXiv](https://arxiv.org/abs/2009.11189): AI quant platform scope: data, learning, backtest, and deployment infrastructure

## Poor Windows

| start | end | strategy | buy-hold | excess | max_dd | avg_pos | bad_score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-02-05 | 2026-05-06 | 24.81% | 211.46% | -186.65pp | 4.52% | 0.147 | 191.17 |
| 2026-02-10 | 2026-05-11 | 23.44% | 200.63% | -177.19pp | 4.52% | 0.147 | 181.70 |
| 2026-02-12 | 2026-05-13 | 22.23% | 191.98% | -169.75pp | 4.52% | 0.147 | 174.26 |
| 2026-02-13 | 2026-05-14 | 21.68% | 188.26% | -166.58pp | 4.52% | 0.147 | 171.10 |
| 2025-04-08 | 2025-07-09 | 63.77% | 223.03% | -159.26pp | 11.50% | 0.466 | 170.76 |
| 2025-04-21 | 2025-07-21 | 60.76% | 215.73% | -154.97pp | 11.50% | 0.530 | 166.47 |
| 2026-02-09 | 2026-05-08 | 19.78% | 176.34% | -156.55pp | 4.52% | 0.147 | 161.07 |
| 2026-02-23 | 2026-05-21 | 19.57% | 170.86% | -151.29pp | 7.26% | 0.147 | 158.55 |
| 2026-02-04 | 2026-05-05 | 18.61% | 171.03% | -152.42pp | 4.52% | 0.147 | 156.94 |
| 2025-04-17 | 2025-07-18 | 57.94% | 197.07% | -139.12pp | 11.50% | 0.524 | 150.62 |
| 2020-03-20 | 2020-06-18 | 57.48% | 188.85% | -131.38pp | 17.17% | 0.499 | 148.55 |
| 2026-02-20 | 2026-05-20 | 17.56% | 158.08% | -140.53pp | 7.26% | 0.147 | 147.79 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| decay_63 | -0.6331 | -0.1781 | -2.833 | 0.065 |
| vol_of_vol_21 | 0.2725 | 0.1045 | 2.695 | -0.029 |
| realized_vol_63 | 1.5829 | 0.9674 | 2.566 | -0.012 |
| tracking_error_vol_21 | 0.1290 | 0.0602 | 1.662 | 0.013 |
| realized_vol_21 | 1.4688 | 0.9644 | 1.630 | -0.007 |
| vol_coef_21 | 1.5999 | 1.0568 | 1.445 | -0.007 |
| tracking_error_21 | -0.0186 | -0.0060 | -1.445 | 0.090 |
| realized_vol_10 | 1.3842 | 0.9587 | 1.101 | -0.004 |
| vix_percentile_252 | 0.7340 | 0.4358 | 0.939 | 0.115 |
| vix_level | 0.5273 | 0.4025 | 0.921 | 0.099 |
| downside_semivol_21 | 0.8909 | 0.6493 | 0.860 | 0.058 |
| jump_variation_5 | 0.8033 | 0.1251 | 0.769 | 0.008 |
| mom_21 | 0.2469 | 0.0553 | 0.762 | -0.183 |
| relative_strength_21 | 1.5811 | 0.3476 | 0.734 | -0.182 |
| mom_10 | 0.1482 | 0.0223 | 0.678 | -0.264 |
| trend_slope_63 | -0.0431 | 0.0449 | -0.671 | 0.007 |
| expectancy_proxy_21 | 0.1862 | 0.0652 | 0.644 | -0.203 |
| rolling_new_low_ratio_63 | 0.0759 | 0.0392 | 0.632 | -0.006 |
| downside_ratio_21 | 0.4188 | 0.4716 | -0.613 | 0.185 |
| vix_zscore | 0.6930 | -0.1244 | 0.612 | 0.107 |
| regime_risk | 0.5650 | 0.4444 | 0.603 | 0.121 |
| sharpe_proxy_21 | 2.8864 | 1.0443 | 0.587 | -0.180 |
| realized_kurtosis_21 | 1.0393 | 0.2857 | 0.559 | -0.031 |
| mom_5 | 0.0810 | 0.0099 | 0.523 | -0.364 |
| trend_gap_21 | 0.0661 | 0.0064 | 0.520 | -0.024 |
| downside_ratio_10 | 0.4010 | 0.4746 | -0.520 | 0.241 |
| anchor_gap_velocity_10 | 0.1010 | -0.0050 | 0.517 | -0.242 |
| realized_skew_21 | 0.2098 | -0.1086 | 0.509 | 0.013 |
| drawdown_63 | -0.3002 | -0.2085 | -0.495 | -0.187 |
| rsi_14 | 0.5973 | 0.5246 | 0.467 | -0.189 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
