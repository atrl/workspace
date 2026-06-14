# Release16 Failed-Fold Participation Target Overlay

Decision: `release16_failedfold_participation_target_overlay_ready_for_checkpoint_no_promotion`.

Source target: `runtime/model_selection/card/rebound_state_factor_target_repair_screen_v1/combined_target_contract.csv` from `rebound_state_factor_target_repair_screen_v1`.
Direct comparison: `release16_failed_fold_daily_path_diagnostic_v1`; reference baseline `soxlbacktest_qlib_walkforward_baseline_v1:wf_pass_ytd18_15_15_02`.

Overlay result:
- wf_2020_train: mean 0.312 -> 0.358; share >=50% 0.246 -> 0.303; share >=65% 0.016 -> 0.299.
- wf_2022_train: mean 0.198 -> 0.198; share >=50% 0.067 -> 0.067; share >=65% 0.044 -> 0.044.
- wf_2023_2024_train: mean 0.155 -> 0.237; share >=50% 0.045 -> 0.300; share >=65% 0.029 -> 0.034.

Readiness checks: `{'target_contract_gate': True, 'wf_2020_overlay_rows_gt0': True, 'wf_2022_guard_unchanged': True, 'wf_2023_2024_overlay_rows_gt0': True}`.
Failed-fold overlay counts: `{'wf_2020': 699, 'wf_2023_2024': 193}`; guard-fold changed rows: `{'wf_2022': 0}`.

Risk note: this card revises only train-window target labels. It does not use test-window labels, train an actor, promote a model, or relax the required walk-forward return/drawdown gates.

Next action: `rerun_release16_checkpoint_screen_with_failedfold_participation_overlay`.

Artifacts: `combined_target_contract.csv`, `target_overlay_rows.csv`, `target_distribution_before_after.csv`, `target_bucket_delta.csv`, `target_scope_audit.csv`, `target_overlay_feature_contrast.csv`, `selection.json`, `summary.json`.
