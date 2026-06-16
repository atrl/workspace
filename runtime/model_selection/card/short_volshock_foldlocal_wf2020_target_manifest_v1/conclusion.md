# short_volshock_foldlocal_wf2020_target_manifest_v1

- decision: `foldlocal_target_coverage_training_ready_official_replay_still_blocks_promotion`
- training_allowed: `True`; bc_screen_allowed_next: `True`
- target_coverage_gate: `True`; target_ready_window_count: `3/3`
- official_replay_coverage_gate: `False`; official_missing_final_required_windows: `wf_2020`
- promotion_allowed, ppo_allowed_next, and actor_retention_allowed remain false.
- next_action: `run_three_fold_short_volshock_bc_slice_screen_before_any_ppo`
