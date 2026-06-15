# Model Retention Frontier

- Rows scanned: `156`.
- Completed handoffs: `120`.
- Recommended next source: `release16_expanded_official_candidate_pool_v1`.
- Recommended next action: `train_nonconvex_checkpoint_objective_with_relative_floor_drawdown_drift_stop_and_sparse_slice_guard_then_official_three_fold_replay`.
- Primary blocker: `walkforward_relative_return_gap`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
