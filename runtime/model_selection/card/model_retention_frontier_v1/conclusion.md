# Model Retention Frontier

- Rows scanned: `157`.
- Completed handoffs: `121`.
- Recommended next source: `release16_nonconvex_checkpoint_objective_score_handoff_v1`.
- Recommended next action: `train_fold_local_nonconvex_checkpoint_actor_with_relative_floor_drawdown_drift_stop_and_sparse_slice_guard_then_official_three_fold_replay`.
- Primary blocker: `training_contract_ready`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
