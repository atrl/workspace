## Conclusion

Decision: `short_volshock_three_fold_bc_slice_ready_for_ppo_screen_no_promotion`.

Selected BC config: `release24_only_steps1000` with weighted fold accuracy `0.807`, min fold accuracy `0.776`, and gate loss `0.0000`.

Three-fold coverage: required folds `wf_2020_train, wf_2022_train, wf_2023_2024_train`; missing `none`. Short-volshock feature folds: `wf_2020_train, wf_2022_train, wf_2023_2024_train`; missing `none`.

Fold fit:
- wf_2020_train: accuracy 0.776, samples 2157, short_target_samples 382, gate True
- wf_2022_train: accuracy 0.913, samples 253, short_target_samples 133, gate True
- wf_2023_2024_train: accuracy 0.862, samples 755, short_target_samples 335, gate True

Required slice fit:
- drawdown_control_state (universal): min accuracy 0.839, min samples 65, gate True
- is_downtrend_state (universal): min accuracy 0.861, min samples 152, gate True
- release_gate_state (universal): min accuracy 0.889, min samples 36, gate True
- short_volshock_state (short_volshock): min accuracy 0.930, min samples 135, gate True
- short_volshock_target_state (short_volshock): min accuracy 0.928, min samples 133, gate True

Gate note: `bc_screen_gate=True`, but `ppo_screen_allowed_next=True` because full short-volshock feature coverage across all required folds is `True`. `promotion_allowed=false` and `actor_retention_allowed=false` because this is only a BC learnability screen; retained-model promotion still requires official replay, MaxDD < 20%, and return alignment against current best and SOXLBacktest baseline.

Next action: `run_short_volshock_walkforward_checkpoint_or_ppo_screen_with_official_replay_gate`.

Artifacts: `combined_target_contract.csv`, `fold_dataset_audit.csv`, `bc_weight_screen.csv`, `fold_bc_metrics.csv`, `state_slice_bc_metrics.csv`, `selection.json`, `summary.json`.
