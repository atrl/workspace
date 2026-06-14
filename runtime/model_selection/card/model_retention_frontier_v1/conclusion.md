# Model Retention Frontier

- Rows scanned: `153`.
- Completed handoffs: `117`.
- Recommended next source: `release16_risk_gated_state_head_objective_screen_v1`.
- Recommended next action: `train_nonconvex_risk_gated_state_head_checkpoint_objective_then_rerun_official_three_fold_replay`.
- Primary blocker: `walkforward_relative_return_gap`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
