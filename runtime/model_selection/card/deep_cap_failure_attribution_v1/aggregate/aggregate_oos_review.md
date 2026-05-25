# Deep-Cap Failure Attribution

## External References

- Qlib: An AI-oriented Quantitative Investment Platform: https://arxiv.org/abs/2009.11189 — research-to-production workflow and persisted quant artifacts
- Probability of Backtest Overfitting: https://doi.org/10.21314/JCF.2016.322 — use OOS/walk-forward evidence before promoting repeated trials
- Focal Loss: https://arxiv.org/abs/1708.02002 — imbalance-aware learning reference already tested in prior rebound-loss screens
- Class-Balanced Loss: https://arxiv.org/abs/1901.05555 — effective-number reweighting reference; do not repeat if failure is not weight-only

## Shared Poor-Window Clusters After Adding Deep-Cap

| cluster | start | end | profiles | type | mean excess | max dd | avg pos | path |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| C002 | 2025-04-04 | 2025-07-22 | 6 | underparticipation | -155.47pp | 16.32% | 0.523 | reentry_participation_or_behavior_clone |
| C003 | 2026-02-04 | 2026-05-21 | 5 | underparticipation | -165.80pp | 24.12% | 0.092 | reentry_participation_or_behavior_clone |
| C001 | 2020-03-18 | 2020-06-19 | 4 | underparticipation | -158.10pp | 13.89% | 0.515 | reentry_participation_or_behavior_clone |

## Top Stable Feature Deltas

| feature | profiles | mean delta_z | consistency | score |
| --- | ---: | ---: | ---: | ---: |
| vol_of_vol_21 | 6 | 2.619 | 1.000 | 5.096 |
| tracking_error_vol_21 | 4 | 3.153 | 1.000 | 5.075 |
| decay_63 | 6 | -2.423 | 1.000 | 4.716 |
| realized_vol_63 | 6 | 2.287 | 1.000 | 4.450 |
| tracking_error_21 | 5 | -1.765 | 1.000 | 3.163 |
| realized_vol_21 | 6 | 1.470 | 1.000 | 2.860 |
| vol_coef_21 | 6 | 1.274 | 1.000 | 2.478 |
| jump_variation_5 | 6 | 1.175 | 1.000 | 2.287 |
| vix_level | 4 | 1.271 | 1.000 | 2.045 |
| realized_vol_10 | 6 | 1.015 | 1.000 | 1.975 |
| vix_percentile_252 | 6 | 0.889 | 1.000 | 1.730 |
| downside_semivol_21 | 5 | 0.904 | 1.000 | 1.619 |

# Gate-Gap Failure Attribution

## Profile Summary

| profile | windows | failures | dominant type | worst window | severity | next mechanism |
| --- | ---: | ---: | --- | --- | ---: | --- |
| deep_capitulation_reentry_guard_v1 | 4 | 4 | drawdown_control | wf_2022 | 217.47 | drawdown_controller_or_sequence_risk_feature |

## Window Attribution

| profile | window | rel return | max dd | avg pos | type | path |
| --- | --- | ---: | ---: | ---: | --- | --- |
| deep_capitulation_reentry_guard_v1 | wf_2022 | 53.59pp | 36.75% | 0.170 | drawdown_control | drawdown_controller_or_sequence_risk_feature |
| deep_capitulation_reentry_guard_v1 | wf_2023_2024 | 36.61pp | 33.78% | 0.485 | drawdown_control | drawdown_controller_or_sequence_risk_feature |
| deep_capitulation_reentry_guard_v1 | strict_oos | 2902.09pp | 32.81% | 0.458 | drawdown_control | drawdown_controller_or_sequence_risk_feature |
| deep_capitulation_reentry_guard_v1 | wf_2020 | 38.49pp | 26.26% | 0.420 | drawdown_control | drawdown_controller_or_sequence_risk_feature |

## Readout

- Treat joint trade-off rows as evidence against another single-coefficient reward sweep.
- Use near-miss drawdown rows for cap calibration or execution friction checks before retraining.
- Use underparticipation or trend-capture rows to justify participation-aware objectives or bull guards.
