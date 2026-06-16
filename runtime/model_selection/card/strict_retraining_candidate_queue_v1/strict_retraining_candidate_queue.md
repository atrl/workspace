# Strict Retraining Candidate Queue

- Decision: `strict_retraining_candidate_queue_ready_no_promotion`
- Final retained models: `0`
- Top candidate: `mode_conditioned_rebound_reward_train_v2`
- Reference baseline: `soxl_backtest_policy_reimplementation_v1`
- Required best return: `213.143462`
- Max WF drawdown: `20.0`

## Queue

| Rank | Candidate | Type | Role | Promotion | Blocker | Evidence |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `mode_conditioned_rebound_reward_train_v2` | trained_actor | train_drawdown_drift_and_return_alignment | False | drawdown_and_return_gap | official_walkforward_gate, worst_wf_maxdd_below_20, ytd_return_above_best_alignment, soxlbacktest_reference_alignment, drawdown_drift_stop_screen |
| 2 | `drawdown_rebound_reward_train_v1` | trained_actor | train_drawdown_drift_and_return_alignment | False | drawdown_and_return_gap | official_walkforward_gate, worst_wf_maxdd_below_20, ytd_return_above_best_alignment, soxlbacktest_reference_alignment, drawdown_drift_stop_screen |
| 3 | `state_conditioned_drawdown_rebound_reward_train_v1` | trained_actor | train_drawdown_drift_and_return_alignment | False | drawdown_and_return_gap | official_walkforward_gate, worst_wf_maxdd_below_20, ytd_return_above_best_alignment, soxlbacktest_reference_alignment, drawdown_drift_stop_screen |
| 4 | `balanced_rule_residual_imitation_v4` | trained_actor | train_drawdown_drift_and_return_alignment | False | drawdown_and_return_gap | official_walkforward_gate, worst_wf_maxdd_below_20, ytd_return_above_best_alignment, soxlbacktest_reference_alignment, drawdown_drift_stop_screen |
| 5 | `balanced_rule_residual_v19` | trained_actor | train_drawdown_drift_and_return_alignment | False | drawdown_and_return_gap | official_walkforward_gate, worst_wf_maxdd_below_20, ytd_return_above_best_alignment, soxlbacktest_reference_alignment, drawdown_drift_stop_screen |
| 6 | `balanced_rule_residual_v19_reg` | trained_actor | train_drawdown_drift_and_return_alignment | False | drawdown_and_return_gap | official_walkforward_gate, worst_wf_maxdd_below_20, ytd_return_above_best_alignment, soxlbacktest_reference_alignment, drawdown_drift_stop_screen |
| 7 | `balanced_rule_residual_v19_reg_light` | trained_actor | train_drawdown_drift_and_return_alignment | False | drawdown_and_return_gap | official_walkforward_gate, worst_wf_maxdd_below_20, ytd_return_above_best_alignment, soxlbacktest_reference_alignment, drawdown_drift_stop_screen |
| 8 | `balanced_rule_residual_v22` | trained_actor | train_drawdown_drift_and_return_alignment | False | drawdown_and_return_gap | official_walkforward_gate, worst_wf_maxdd_below_20, ytd_return_above_best_alignment, soxlbacktest_reference_alignment, drawdown_drift_stop_screen |
| 9 | `smoke` | trained_actor | complete_official_walkforward_before_actor_retention | False | missing_wf_windows | complete_missing_wf_windows, official_walkforward_gate, worst_wf_maxdd_below_20, ytd_return_above_best_alignment, soxlbacktest_reference_alignment |
| 10 | `balanced_macro_v1` | trained_actor | complete_official_walkforward_before_actor_retention | False | return_gap | complete_missing_wf_windows, official_walkforward_gate, worst_wf_maxdd_below_20, ytd_return_above_best_alignment, soxlbacktest_reference_alignment |
| 11 | `balanced_rule_residual_v19_reg_v22_retest_v1` | trained_actor | complete_official_walkforward_before_actor_retention | False | return_gap | complete_missing_wf_windows, official_walkforward_gate, worst_wf_maxdd_below_20, ytd_return_above_best_alignment, soxlbacktest_reference_alignment |
| 12 | `balanced` | trained_actor | complete_official_walkforward_before_actor_retention | False | return_gap | complete_missing_wf_windows, official_walkforward_gate, worst_wf_maxdd_below_20, ytd_return_above_best_alignment, soxlbacktest_reference_alignment |
| 13 | `soxl_backtest_policy_reimplementation_v1` | reference_baseline | buy_sell_reference_baseline | False | reference_baseline_not_actor | compare_buy_sell_features, common_window_rebased_nav, reference_only_not_actor_retention |
| 14 | `short_foldlocal_drawdown_drift_training_screen` | training_screen | screen_only_no_promotion | False | must_pass_official_walkforward_gate_after_screen | foldlocal_screen_metrics, official_walkforward_gate, final_walkforward_retention_contract |

## Required Gates

| Gate | Required | Current | Status |
| --- | ---: | ---: | --- |
| `final_walkforward_retention_contract` | True | 0 | blocked |
| `strict_walkforward_profile_count` | 1 | 0 | blocked |
| `max_walkforward_drawdown_pct` | 20.0 | 21.83172 | blocked |
| `required_best_return_pct` | 213.143462 | 177.573569 | blocked |

## Conclusion

No actor can be retained until the final walk-forward contract passes. Use the first trained-actor rows as diagnostic retraining targets and keep SOXLBacktest as a reference baseline only.
