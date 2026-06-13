## Conclusion

Decision: `three_fold_bc_slice_ready_for_checkpoint_screen_no_promotion`.

Selected BC config: `release10_rebound4_downtrend2_steps600` with weighted fold accuracy `0.759`, min fold accuracy `0.727`, and gate loss `0.0040`.

Three-fold coverage: required folds `wf_2020_train, wf_2022_train, wf_2023_2024_train`; missing `none`; `wf_2020_coverage=True`.

Fold fit:
- wf_2020_train: accuracy 0.727, samples 2157, gate True
- wf_2022_train: accuracy 0.824, samples 505, gate True
- wf_2023_2024_train: accuracy 0.807, samples 755, gate True

Required sparse-state slice fit:
- release_gate_state: min accuracy 0.611, min samples 36, gate True
- is_downtrend_state: min accuracy 0.833, min samples 234, gate True
- drawdown_control_state: min accuracy 0.870, min samples 49, gate True

Risk and promotion note: this is only a BC learnability screen. `promotion_allowed=false`, `actor_retention_allowed=false`, and `training_allowed_from_this_card=false`; a later checkpoint/PPO screen is allowed only when `bc_screen_gate=true`, and runtime promotion still requires fixed-split plus `wf_2020`, `wf_2022`, and `wf_2023_2024` replay against excess >= +30pp and MaxDD <= 15%.

Next action: `run_walkforward_checkpoint_or_ppo_screen_with_bc_warmstart_against_soxlbacktest_reference_and_drawdown_drift_stop_rules`.

Artifacts: `combined_target_contract.csv`, `bc_weight_screen.csv`, `state_slice_bc_metrics.csv`, `fold_bc_metrics.csv`, `bc_slice_summary.json`, `selection.json`, `summary.json`.
