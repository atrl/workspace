# Strict Candidate Retention Filter

- decision: `strict_candidate_retention_filter_no_promotable_candidates`
- reference_baseline: `soxl_backtest_policy_reimplementation_v1`
- promotable_profile_count: `0`
- actor_retention_allowed_count: `0`
- ppo_allowed_next: `False`

## Candidate Rows

- `soxl_backtest_policy_reimplementation_v1` (reference_baseline): promotion=False, actor_retention=False, blocker=`reference_baseline_comparison_only`
- `balanced` (profile): promotion=False, actor_retention=False, blocker=`missing_wf_window:wf_2020`
- `balanced_macro_v1` (profile): promotion=False, actor_retention=False, blocker=`missing_wf_window:wf_2020`
- `balanced_rule_residual_v19` (profile): promotion=False, actor_retention=False, blocker=`wf_max_drawdown_ge_20:wf_2020`
- `smoke` (profile): promotion=False, actor_retention=False, blocker=`missing_wf_window:wf_2020`
- `balanced_rule_residual_v22` (profile): promotion=False, actor_retention=False, blocker=`wf_max_drawdown_ge_20:wf_2020`
- `balanced_rule_residual_imitation_v4` (profile): promotion=False, actor_retention=False, blocker=`wf_max_drawdown_ge_20:wf_2020`
- `drawdown_rebound_reward_train_v1` (profile): promotion=False, actor_retention=False, blocker=`wf_max_drawdown_ge_20:wf_2020`
- `mode_conditioned_rebound_reward_train_v2` (profile): promotion=False, actor_retention=False, blocker=`wf_max_drawdown_ge_20:wf_2020`
- `state_conditioned_drawdown_rebound_reward_train_v1` (profile): promotion=False, actor_retention=False, blocker=`wf_max_drawdown_ge_20:wf_2020`
- `balanced_rule_residual_v19_reg` (profile): promotion=False, actor_retention=False, blocker=`wf_max_drawdown_ge_20:wf_2022`
- `balanced_rule_residual_v19_reg_light` (profile): promotion=False, actor_retention=False, blocker=`wf_max_drawdown_ge_20:wf_2020`
- `balanced_rule_residual_v19_reg_v22_retest_v1` (profile): promotion=False, actor_retention=False, blocker=`missing_wf_window:wf_2020`
- `soxlbacktest_qlib_walkforward_baseline_v1` (qlib_reference_screen): promotion=False, actor_retention=False, blocker=`soxlbacktest_reference_qlib_aligned_not_promotable`
- `cap10_three_fold_checkpoint_drift_selector_v1` (checkpoint_selector): promotion=False, actor_retention=False, blocker=`trend_underparticipation`
