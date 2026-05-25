# OOS Poor-Window Diagnostic: cap10_release_0.10__wf_2023_2024

## Scope

- train_end: `2021-12-31`
- OOS window: `2023-01-01` .. `2024-12-31`
- target_excess=30.00%
- target_max_dd=15.00%

## External References

- [Qlib GitHub](https://github.com/microsoft/qlib): keep the replay/backtest workflow inside the qlib harness
- [Qlib arXiv](https://arxiv.org/abs/2009.11189): AI quant platform scope for data, modeling, backtest, and deployment
- [Probability of Backtest Overfitting](https://doi.org/10.21314/JCF.2016.322): avoid selecting a candidate from repeated screens without OOS discipline
- [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551): multiple-testing caution for model and factor screens

## Poor Windows

| start | end | strategy | buy-hold | excess | max_dd | avg_pos | bad_score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023-12-06 | 2024-03-07 | 27.12% | 157.66% | -130.54pp | 5.15% | 0.085 | 160.54 |
| 2023-10-25 | 2024-01-25 | 19.68% | 147.96% | -128.28pp | 5.15% | 0.115 | 158.28 |
| 2023-10-27 | 2024-01-29 | 18.89% | 132.05% | -113.16pp | 5.15% | 0.112 | 143.16 |
| 2023-10-26 | 2024-01-26 | 18.81% | 130.89% | -112.08pp | 5.15% | 0.114 | 142.08 |
| 2023-10-30 | 2024-01-30 | 18.80% | 130.76% | -111.97pp | 5.15% | 0.110 | 141.97 |
| 2023-12-05 | 2024-03-06 | 24.67% | 129.00% | -104.33pp | 5.15% | 0.086 | 134.33 |
| 2023-10-23 | 2024-01-23 | 17.05% | 119.16% | -102.10pp | 5.15% | 0.117 | 132.10 |
| 2023-10-24 | 2024-01-24 | 18.32% | 119.60% | -101.28pp | 5.15% | 0.117 | 131.28 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| rolling_new_low_ratio_63 | 0.0870 | 0.0230 | 2.271 | -0.079 |
| realized_vol_63 | 0.7635 | 0.9836 | -1.134 | 0.011 |
| expectancy_proxy_21 | 0.1889 | 0.0361 | 0.960 | -0.181 |
| sharpe_proxy_21 | 3.1465 | 0.6186 | 0.887 | -0.173 |
| cum_drop_30 | -0.0707 | -0.1692 | 0.826 | -0.296 |
| drawdown_63 | -0.0990 | -0.2207 | 0.819 | -0.212 |
| vix_level | 0.2808 | 0.3349 | -0.807 | 0.145 |
| downside_semivol_21 | 0.4466 | 0.6226 | -0.753 | 0.107 |
| vol_stress_score | 0.6673 | 0.4433 | 0.739 | -0.061 |
| negative_gap_frequency_21 | 0.2938 | 0.3643 | -0.732 | 0.075 |
| decay_63 | -0.2729 | -0.3212 | 0.720 | 0.152 |
| rsi_14 | 0.6140 | 0.5095 | 0.713 | -0.204 |
| trend_slope_21 | 0.0808 | 0.0096 | 0.686 | 0.030 |
| mom_21 | 0.2035 | 0.0473 | 0.681 | -0.170 |
| relative_strength_21 | 1.3228 | 0.2799 | 0.681 | -0.167 |
| trend_gap_63 | 0.1188 | 0.0266 | 0.660 | 0.026 |
| drawdown_21 | -0.0695 | -0.1422 | 0.658 | -0.324 |
| cum_drop_20 | -0.0691 | -0.1386 | 0.632 | -0.328 |
| vol_of_vol_21 | 0.0672 | 0.1072 | -0.606 | -0.028 |
| trend_gap_21 | 0.0694 | 0.0058 | 0.602 | 0.000 |
| days_since_high_63 | 0.2376 | 0.4187 | -0.595 | 0.010 |
| drawdown_streak_63 | 0.2622 | 0.4700 | -0.583 | 0.003 |
| downside_ratio_21 | 0.4382 | 0.4799 | -0.561 | 0.122 |
| realized_vol_21 | 0.7902 | 0.9356 | -0.547 | -0.002 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
