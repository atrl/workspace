# Model Retention Frontier

- Rows scanned: `152`.
- Completed handoffs: `116`.
- Recommended next source: `release16_state_head_revision_handoff_v1`.
- Recommended next action: `build_risk_gated_state_head_target_or_drawdown_drift_checkpoint_objective_then_rerun_official_three_fold_replay`.
- Primary blocker: `walkforward_relative_return_gap`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
