# Final Walk-Forward Retention Contract

- decision: `final_walkforward_retention_contract_no_retained_models`
- final_retained_count: `0`
- reference_baseline: `soxl_backtest_policy_reimplementation_v1`
- reference_is_current: `True`
- strict_passing_profile_count: `0`

## Rows

- `soxl_backtest_policy_reimplementation_v1`: final_retained=False, actor_retention=False, blockers=`SOXLBacktest reference baseline cannot be retained as actor`
- `balanced`: final_retained=False, actor_retention=False, blockers=`missing_wf_window:wf_2020; missing_wf_window:wf_2022; missing_wf_window:wf_2023_2024; ytd_return_below_best_alignment`
- `balanced_macro_v1`: final_retained=False, actor_retention=False, blockers=`missing_wf_window:wf_2020; missing_wf_window:wf_2022; missing_wf_window:wf_2023_2024; ytd_return_below_best_alignment`
- `balanced_rule_residual_v19`: final_retained=False, actor_retention=False, blockers=`wf_max_drawdown_ge_20:wf_2020; wf_max_drawdown_ge_20:wf_2022; wf_max_drawdown_ge_20:wf_2023_2024; ytd_return_below_best_alignment`
- `smoke`: final_retained=False, actor_retention=False, blockers=`missing_wf_window:wf_2020; missing_wf_window:wf_2022; missing_wf_window:wf_2023_2024`
- `balanced_rule_residual_v22`: final_retained=False, actor_retention=False, blockers=`wf_max_drawdown_ge_20:wf_2020; wf_max_drawdown_ge_20:wf_2022; wf_max_drawdown_ge_20:wf_2023_2024; ytd_return_below_soxlbacktest_baseline; ytd_return_below_best_alignment`
- `balanced_rule_residual_imitation_v4`: final_retained=False, actor_retention=False, blockers=`wf_max_drawdown_ge_20:wf_2020; wf_max_drawdown_ge_20:wf_2022; wf_max_drawdown_ge_20:wf_2023_2024; ytd_return_below_soxlbacktest_baseline; ytd_return_below_best_alignment`
- `drawdown_rebound_reward_train_v1`: final_retained=False, actor_retention=False, blockers=`wf_max_drawdown_ge_20:wf_2020; ytd_return_below_best_alignment`
- `mode_conditioned_rebound_reward_train_v2`: final_retained=False, actor_retention=False, blockers=`wf_max_drawdown_ge_20:wf_2020; wf_max_drawdown_ge_20:wf_2023_2024; ytd_return_below_best_alignment`
- `state_conditioned_drawdown_rebound_reward_train_v1`: final_retained=False, actor_retention=False, blockers=`wf_max_drawdown_ge_20:wf_2020; ytd_return_below_best_alignment`
- `balanced_rule_residual_v19_reg`: final_retained=False, actor_retention=False, blockers=`wf_max_drawdown_ge_20:wf_2022; wf_max_drawdown_ge_20:wf_2023_2024; ytd_return_below_soxlbacktest_baseline; ytd_return_below_best_alignment`
- `balanced_rule_residual_v19_reg_light`: final_retained=False, actor_retention=False, blockers=`wf_max_drawdown_ge_20:wf_2020; wf_max_drawdown_ge_20:wf_2022; wf_max_drawdown_ge_20:wf_2023_2024; ytd_return_below_best_alignment`
- `balanced_rule_residual_v19_reg_v22_retest_v1`: final_retained=False, actor_retention=False, blockers=`missing_wf_window:wf_2020; missing_wf_window:wf_2022; missing_wf_window:wf_2023_2024; ytd_return_below_best_alignment`
