# OOS Poor-Window Diagnostic: base_multi_obj_plain__wf_2022

## Scope

- train_end: `fold-local replay source`
- OOS window: `2022-01-03` .. `2022-12-29`
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
| 2022-10-14 | 2022-12-13 | 8.94% | 96.68% | -87.74pp | 10.79% | 0.284 | 117.74 |
| 2022-10-17 | 2022-12-14 | 6.61% | 77.35% | -70.74pp | 10.79% | 0.284 | 100.74 |
| 2022-10-12 | 2022-12-09 | 5.62% | 67.85% | -62.23pp | 10.79% | 0.284 | 92.23 |
| 2022-10-11 | 2022-12-08 | 5.44% | 66.71% | -61.27pp | 10.79% | 0.284 | 91.27 |
| 2022-10-13 | 2022-12-12 | 4.80% | 62.25% | -57.45pp | 10.79% | 0.284 | 87.45 |
| 2022-10-18 | 2022-12-15 | 3.99% | 54.05% | -50.07pp | 10.79% | 0.286 | 80.07 |
| 2022-10-19 | 2022-12-16 | 3.43% | 46.76% | -43.34pp | 10.79% | 0.289 | 73.34 |
| 2022-06-16 | 2022-08-16 | 12.79% | 55.78% | -42.99pp | 9.47% | 0.235 | 72.99 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| expectancy_proxy_21 | 0.0548 | -0.1119 | 1.238 | -0.236 |
| rolling_new_low_ratio_63 | 0.1679 | 0.1025 | 1.192 | -0.093 |
| mom_21 | 0.0861 | -0.1673 | 1.184 | -0.219 |
| relative_strength_21 | 0.5540 | -1.1265 | 1.180 | -0.218 |
| sharpe_proxy_21 | 0.8045 | -1.7037 | 1.130 | -0.212 |
| vix_percentile_252 | 0.5362 | 0.7625 | -1.068 | 0.169 |
| regime_risk | 0.5335 | 0.6756 | -1.058 | 0.174 |
| trend_gap_21 | 0.0142 | -0.0905 | 1.045 | -0.020 |
| rsi_14 | 0.5218 | 0.4067 | 1.014 | -0.229 |
| trend_slope_21 | 0.0003 | -0.0882 | 0.998 | 0.068 |
| downside_ratio_21 | 0.4783 | 0.5452 | -0.951 | 0.233 |
| anchor_slope_180 | -0.1251 | -0.0488 | -0.942 | 0.039 |
| vol_coef_21 | 1.1329 | 1.4480 | -0.901 | 0.007 |
| realized_kurtosis_21 | 0.3589 | -0.3933 | 0.892 | 0.000 |
| regime_risk_velocity_5 | -0.0424 | 0.0201 | -0.876 | 0.366 |
| downside_semivol_21 | 0.8318 | 0.9502 | -0.828 | 0.109 |
| vix_zscore | 0.1204 | 1.0374 | -0.824 | 0.154 |
| mom_10 | 0.0587 | -0.0978 | 0.813 | -0.282 |
| regime_risk_no_vix | 0.5257 | 0.6289 | -0.812 | 0.113 |
| anchor_gap_velocity_10 | 0.0468 | -0.0768 | 0.809 | -0.271 |
| trend_slope_63 | -0.1756 | -0.0919 | -0.789 | 0.112 |
| below_anchor_ratio_63 | 1.0000 | 0.7212 | 0.775 | -0.056 |
| days_since_high_63 | 0.8654 | 0.6756 | 0.763 | -0.094 |
| drawdown_21 | -0.1859 | -0.2789 | 0.746 | -0.296 |
| cum_drop_20 | -0.1800 | -0.2716 | 0.742 | -0.303 |
| cum_drop_30 | -0.2364 | -0.3339 | 0.737 | -0.203 |
| tracking_error_vol_21 | 0.0799 | 0.0679 | 0.718 | -0.012 |
| rsi_6 | 0.5400 | 0.3950 | 0.716 | -0.627 |
| downside_ratio_10 | 0.4767 | 0.5519 | -0.647 | 0.241 |
| anchor_gap_180 | -0.5056 | -0.3402 | -0.641 | -0.075 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
