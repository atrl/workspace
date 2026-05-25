# OOS Poor-Window Diagnostic: mode_conditioned_rebound_reward_train_v2

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
| 2025-04-08 | 2025-07-09 | 30.64% | 223.03% | -192.39pp | 6.58% | 0.458 | 198.97 |
| 2026-02-05 | 2026-05-06 | 40.32% | 211.46% | -171.14pp | 16.76% | 0.065 | 187.90 |
| 2025-04-21 | 2025-07-21 | 35.51% | 215.73% | -180.22pp | 3.92% | 0.460 | 184.14 |
| 2026-02-10 | 2026-05-11 | 35.71% | 200.63% | -164.92pp | 16.76% | 0.043 | 181.69 |
| 2020-03-20 | 2020-06-18 | 16.34% | 188.85% | -172.51pp | 5.90% | 0.389 | 178.41 |
| 2026-02-12 | 2026-05-13 | 34.40% | 191.98% | -157.58pp | 16.76% | 0.033 | 174.35 |
| 2026-02-13 | 2026-05-14 | 33.88% | 188.26% | -154.38pp | 16.76% | 0.028 | 171.14 |
| 2025-04-17 | 2025-07-18 | 31.54% | 197.07% | -165.53pp | 3.92% | 0.457 | 169.45 |
| 2020-03-18 | 2020-06-16 | 15.47% | 177.82% | -162.35pp | 5.90% | 0.383 | 168.25 |
| 2025-04-07 | 2025-07-08 | 28.69% | 188.20% | -159.51pp | 6.58% | 0.458 | 166.09 |
| 2025-04-16 | 2025-07-17 | 31.12% | 192.51% | -161.38pp | 3.92% | 0.457 | 165.30 |
| 2025-04-04 | 2025-07-07 | 27.33% | 186.03% | -158.70pp | 6.58% | 0.458 | 165.28 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| decay_63 | -0.6511 | -0.1761 | -3.103 | 0.062 |
| vol_of_vol_21 | 0.2782 | 0.1039 | 2.927 | -0.034 |
| tracking_error_vol_21 | 0.1378 | 0.0590 | 2.698 | 0.012 |
| realized_vol_63 | 1.5913 | 0.9671 | 2.618 | -0.017 |
| tracking_error_21 | -0.0201 | -0.0058 | -1.946 | 0.090 |
| realized_vol_21 | 1.4851 | 0.9627 | 1.727 | -0.008 |
| vol_coef_21 | 1.6271 | 1.0535 | 1.578 | -0.009 |
| realized_vol_10 | 1.4047 | 0.9563 | 1.199 | -0.005 |
| vix_level | 0.5453 | 0.4000 | 1.139 | 0.098 |
| downside_semivol_21 | 0.9153 | 0.6460 | 0.993 | 0.060 |
| vix_percentile_252 | 0.7434 | 0.4348 | 0.976 | 0.117 |
| jump_variation_5 | 0.8970 | 0.1122 | 0.961 | -0.001 |
| trend_slope_63 | -0.0565 | 0.0467 | -0.777 | 0.011 |
| vix_zscore | 0.8207 | -0.1422 | 0.745 | 0.108 |
| rolling_new_low_ratio_63 | 0.0789 | 0.0388 | 0.693 | -0.011 |
| mom_21 | 0.2160 | 0.0601 | 0.629 | -0.187 |
| regime_risk | 0.5690 | 0.4440 | 0.629 | 0.124 |
| realized_kurtosis_21 | 1.0907 | 0.2792 | 0.602 | -0.021 |
| tracking_error_5 | -0.0185 | -0.0061 | -0.599 | 0.132 |
| relative_strength_21 | 1.3700 | 0.3807 | 0.599 | -0.186 |
| mom_10 | 0.1307 | 0.0251 | 0.585 | -0.270 |
| drawdown_63 | -0.3134 | -0.2067 | -0.584 | -0.194 |
| expectancy_proxy_21 | 0.1722 | 0.0675 | 0.559 | -0.205 |
| downside_ratio_21 | 0.4249 | 0.4706 | -0.530 | 0.182 |
| sharpe_proxy_21 | 2.6806 | 1.0774 | 0.513 | -0.183 |
| realized_skew_21 | 0.2077 | -0.1078 | 0.506 | 0.013 |
| mom_5 | 0.0767 | 0.0106 | 0.493 | -0.373 |
| below_anchor_ratio_63 | 0.5538 | 0.3526 | 0.491 | 0.012 |
| anchor_gap_velocity_10 | 0.0946 | -0.0039 | 0.485 | -0.250 |
| downside_ratio_10 | 0.4063 | 0.4737 | -0.477 | 0.241 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
