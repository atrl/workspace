# Walk-Forward Weighted Objective Selection

## Decision

- decision: `no_full_gate_candidate`
- selected_candidate: `recovery_reentry_conditional_vol_target_v1`
- next_mechanism: `walk_forward_weighted_multi_objective_training`

## Candidate Ranking

| candidate | full | loss | pass/windows | min rel | max dd | failure | next |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| recovery_reentry_conditional_vol_target_v1 | False | 237.50 | 0/3 | 55.80pp | 26.14% | drawdown_control | drawdown_state_guard_or_sequence_risk_feature |
| recovery_reentry_cppi_floor_v1 | False | 276.14 | 0/3 | 55.80pp | 30.18% | drawdown_control | drawdown_state_guard_or_sequence_risk_feature |
| recovery_reentry_regime_drawdown_guard_v1 | False | 378.45 | 0/3 | 55.80pp | 30.18% | drawdown_control | drawdown_state_guard_or_sequence_risk_feature |
| recovery_reentry_2021_trendbreak_cap_v1 | False | 384.59 | 0/3 | 25.56pp | 28.20% | drawdown_control | drawdown_state_guard_or_sequence_risk_feature |
| soxl_soxx_efficiency_chop_guard_v1 | False | 385.93 | 0/3 | 25.82pp | 30.51% | drawdown_control | drawdown_state_guard_or_sequence_risk_feature |
| recovery_reentry_floor_v1 | False | 401.92 | 0/3 | 49.90pp | 30.18% | drawdown_control | drawdown_state_guard_or_sequence_risk_feature |
| recovery_reentry_anchor_confirm_v1 | False | 408.86 | 0/3 | 55.80pp | 30.18% | drawdown_control | drawdown_state_guard_or_sequence_risk_feature |
| v19reg_v22_budget_usage_bull_guard_v1 | False | 408.86 | 0/3 | 40.18pp | 30.18% | drawdown_control | drawdown_state_guard_or_sequence_risk_feature |
| recovery_reentry_equity_drawdown_guard_v1 | False | 420.48 | 0/3 | -15.41pp | 28.25% | drawdown_control | drawdown_state_guard_or_sequence_risk_feature |
| v19reglight_v22_budget_usage_bull_guard_ablation_v1 | False | 452.68 | 0/3 | 33.69pp | 36.91% | drawdown_control | drawdown_state_guard_or_sequence_risk_feature |
| v19reg_v22_budget_usage_v1 | False | 484.23 | 0/3 | -8.23pp | 30.18% | drawdown_control | drawdown_state_guard_or_sequence_risk_feature |
| balanced_rule_residual_mode_v1 | False | 695.25 | 0/3 | -14.57pp | 41.02% | joint_tradeoff | walk_forward_weighted_multi_objective_training |
| balanced_rule_residual_imitation_v4 | False | 804.79 | 0/3 | -12.01pp | 43.75% | drawdown_control | drawdown_state_guard_or_sequence_risk_feature |
| risk_budget_residual_drawdown_guard_walkforward_v1 | False | 855.03 | 3/12 | -54.19pp | 28.57% | joint_tradeoff | walk_forward_weighted_multi_objective_training |

## Readout

- A full-gate candidate must satisfy every supplied window under +30pp and <=15% MaxDD.
- Lower weighted loss is diagnostic only when no candidate fully passes.
- Keep this selector out of runtime auto-selection until a notebook promotes it with replay evidence.
