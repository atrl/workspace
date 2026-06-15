# Model Retention Frontier

- Rows scanned: `155`.
- Completed handoffs: `119`.
- Recommended next source: `release16_nonconvex_checkpoint_training_contract_v1`.
- Recommended next action: `train_nonconvex_checkpoint_objective_with_release_slice_guard_and_drawdown_drift_stop_then_official_three_fold_replay`.
- Primary blocker: `training_contract_ready`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
