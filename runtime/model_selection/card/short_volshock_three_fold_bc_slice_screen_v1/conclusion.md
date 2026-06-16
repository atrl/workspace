## Conclusion

Decision: `short_volshock_three_fold_bc_slice_failed_no_ppo`.

Selected BC config: `release10_downtrend3_obj25_cb99_steps600` with weighted fold accuracy `0.704`, min fold accuracy `0.637`, and gate loss `0.5635`.

Three-fold coverage: required folds `wf_2020_train, wf_2022_train, wf_2023_2024_train`; missing `none`. Short-volshock feature folds: `wf_2022_train, wf_2023_2024_train`; missing `wf_2020_train`.

Fold fit:
- wf_2020_train: accuracy 0.637, samples 2157, short_target_samples 0, gate False
- wf_2022_train: accuracy 0.937, samples 253, short_target_samples 133, gate True
- wf_2023_2024_train: accuracy 0.817, samples 755, short_target_samples 335, gate True

Required slice fit:
- drawdown_control_state (universal): min accuracy 0.880, min samples 65, gate True
- is_downtrend_state (universal): min accuracy 0.880, min samples 152, gate True
- release_gate_state (universal): min accuracy 0.639, min samples 36, gate True
- short_volshock_state (short_volshock): min accuracy 0.947, min samples 135, gate True
- short_volshock_target_state (short_volshock): min accuracy 0.946, min samples 133, gate True

Gate note: `bc_screen_gate=False`, but `ppo_screen_allowed_next=False` because full short-volshock feature coverage across all required folds is `False`. `promotion_allowed=false` and `actor_retention_allowed=false` because this is only a BC learnability screen; retained-model promotion still requires official replay, MaxDD < 20%, and return alignment against current best and SOXLBacktest baseline.

Next action: `retune_short_volshock_targets_or_weighting_before_any_ppo`.

Artifacts: `combined_target_contract.csv`, `fold_dataset_audit.csv`, `bc_weight_screen.csv`, `fold_bc_metrics.csv`, `state_slice_bc_metrics.csv`, `selection.json`, `summary.json`.
