# OOS Poor-Window Diagnostic: cap10_release_0.30__wf_2022

## Scope

- train_end: `2020-12-31`
- OOS window: `2022-01-01` .. `2022-12-31`
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
| 2022-09-15 | 2022-12-13 | -3.45% | 14.44% | -17.89pp | 6.78% | 0.019 | 47.89 |
| 2022-09-23 | 2022-12-21 | -3.65% | 11.07% | -14.71pp | 6.78% | 0.009 | 44.71 |
| 2022-09-30 | 2022-12-29 | -4.67% | 9.14% | -13.81pp | 8.08% | 0.002 | 43.81 |
| 2022-09-16 | 2022-12-14 | -3.78% | 7.79% | -11.58pp | 6.78% | 0.017 | 41.58 |
| 2022-09-06 | 2022-12-02 | -3.40% | 4.91% | -8.31pp | 6.76% | 0.044 | 38.31 |
| 2022-09-14 | 2022-12-12 | -3.97% | 4.17% | -8.14pp | 6.78% | 0.020 | 38.14 |
| 2022-09-02 | 2022-12-01 | -2.44% | 4.98% | -7.41pp | 6.76% | 0.048 | 37.41 |
| 2022-09-26 | 2022-12-22 | -4.56% | 2.41% | -6.97pp | 7.21% | 0.008 | 36.97 |

## Feature Differences

| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |
| --- | ---: | ---: | ---: | ---: |
| realized_kurtosis_21 | 0.8931 | -0.6151 | 3.404 | 0.000 |
| anchor_slope_180 | -0.1667 | -0.0329 | -2.349 | 0.048 |
| vol_coef_21 | 1.0455 | 1.4712 | -1.378 | 0.013 |
| regime_risk_no_vix | 0.4912 | 0.6386 | -1.274 | 0.115 |
| regime_risk | 0.5120 | 0.6770 | -1.185 | 0.169 |
| tracking_error_vol_21 | 0.0832 | 0.0670 | 1.172 | -0.021 |
| new_low_ratio_21 | 0.2631 | 0.1637 | 1.097 | -0.016 |
| vix_percentile_252 | 0.5336 | 0.7497 | -1.093 | 0.160 |
| decay_63 | -0.1483 | -0.0899 | -0.913 | 0.141 |
| vix_zscore | 0.1243 | 0.9804 | -0.786 | 0.144 |
| below_ma63_days | 0.2772 | 0.6004 | -0.772 | -0.009 |
| below_anchor_ratio_63 | 1.0000 | 0.7361 | 0.743 | -0.060 |
| realized_skew_21 | 0.2368 | 0.0809 | 0.671 | 0.056 |
| drawdown_streak_63 | 1.0000 | 0.8393 | 0.592 | -0.026 |
| vol_stress_score | 0.4392 | 0.5753 | -0.555 | 0.007 |
| anchor_gap_180 | -0.4878 | -0.3569 | -0.500 | -0.067 |
| tracking_error_21 | -0.0055 | -0.0035 | -0.478 | 0.172 |
| vol_of_vol_21 | 0.1203 | 0.0997 | 0.457 | -0.026 |
| anchor_gap_velocity_10 | 0.0173 | -0.0566 | 0.453 | -0.267 |
| jump_variation_5 | 0.5760 | 0.0000 | 0.436 | -0.066 |
| downside_semivol_21 | 0.8612 | 0.9292 | -0.350 | 0.102 |
| trend_gap_63 | -0.1059 | -0.1471 | 0.325 | 0.131 |
| loss_streak_21 | 0.0621 | 0.0459 | 0.287 | 0.779 |
| intraday_range | 0.0932 | 0.1040 | -0.286 | 0.198 |

## Next Actions

- Treat repeated negative-excess windows as overfit candidates before retraining.
- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.
- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.
