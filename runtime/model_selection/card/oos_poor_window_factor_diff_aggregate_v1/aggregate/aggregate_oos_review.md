# Aggregated OOS Poor-Window Diagnostic

## Shared Bad-Window Clusters

| cluster | start | end | profiles | type | mean excess | max dd | avg pos | next path |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| C002 | 2025-04-04 | 2025-07-22 | 5 | underparticipation | -160.50pp | 16.32% | 0.520 | reentry_participation_or_behavior_clone |
| C003 | 2026-02-04 | 2026-05-21 | 4 | underparticipation | -167.30pp | 24.12% | 0.089 | reentry_participation_or_behavior_clone |
| C001 | 2020-03-18 | 2020-06-19 | 4 | underparticipation | -158.10pp | 13.89% | 0.515 | reentry_participation_or_behavior_clone |

## Stable Feature Deltas

| feature | profiles | mean delta_z | consistency | mean spearman | score |
| --- | ---: | ---: | ---: | ---: | ---: |
| vol_of_vol_21 | 5 | 2.909 | 1.000 | -0.029 | 5.212 |
| tracking_error_vol_21 | 4 | 3.153 | 1.000 | 0.008 | 5.075 |
| decay_63 | 5 | -2.832 | 1.000 | 0.065 | 5.074 |
| realized_vol_63 | 5 | 2.528 | 1.000 | -0.015 | 4.530 |
| tracking_error_21 | 4 | -2.162 | 1.000 | 0.091 | 3.479 |
| realized_vol_21 | 5 | 1.624 | 1.000 | -0.003 | 2.909 |
| vol_coef_21 | 5 | 1.465 | 1.000 | -0.003 | 2.624 |
| vix_level | 4 | 1.271 | 1.000 | 0.097 | 2.045 |
| jump_variation_5 | 5 | 1.126 | 1.000 | 0.005 | 2.017 |
| realized_vol_10 | 5 | 1.112 | 1.000 | -0.002 | 1.992 |
| downside_semivol_21 | 4 | 1.074 | 1.000 | 0.060 | 1.729 |
| vix_percentile_252 | 5 | 0.921 | 1.000 | 0.116 | 1.650 |
| trend_slope_63 | 4 | -0.990 | 1.000 | 0.013 | 1.594 |
| rolling_new_low_ratio_63 | 5 | 0.766 | 1.000 | -0.011 | 1.373 |
| realized_kurtosis_21 | 5 | 0.728 | 1.000 | -0.023 | 1.304 |
| vix_zscore | 5 | 0.704 | 1.000 | 0.107 | 1.261 |
| drawdown_63 | 4 | -0.720 | 1.000 | -0.186 | 1.159 |
| realized_skew_21 | 5 | 0.631 | 1.000 | 0.010 | 1.130 |
| below_anchor_ratio_63 | 4 | 0.686 | 1.000 | 0.007 | 1.104 |
| tracking_error_5 | 4 | -0.658 | 1.000 | 0.134 | 1.060 |
| mom_21 | 4 | 0.645 | 1.000 | -0.185 | 1.038 |
| regime_risk | 4 | 0.636 | 1.000 | 0.123 | 1.023 |
| relative_strength_21 | 4 | 0.619 | 1.000 | -0.183 | 0.997 |
| expectancy_proxy_21 | 4 | 0.598 | 1.000 | -0.204 | 0.962 |
| mom_10 | 4 | 0.576 | 1.000 | -0.268 | 0.928 |
| downside_ratio_21 | 4 | -0.575 | 1.000 | 0.184 | 0.925 |
| sharpe_proxy_21 | 4 | 0.544 | 1.000 | -0.182 | 0.876 |
| downside_ratio_10 | 4 | -0.498 | 1.000 | 0.243 | 0.802 |
| mom_5 | 4 | 0.470 | 1.000 | -0.369 | 0.757 |
| anchor_gap_velocity_10 | 3 | 0.518 | 1.000 | -0.251 | 0.718 |

## Readout

- Use shared clusters to decide whether the next loop is participation, drawdown control, or model selection.
- Treat feature deltas as mechanism hypotheses until they pass IC, single-signal, and walk-forward gates.
