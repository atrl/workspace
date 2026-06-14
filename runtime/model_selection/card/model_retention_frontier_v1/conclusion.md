# Model Retention Frontier

- Rows scanned: `105`.
- Completed handoffs: `71`.
- Recommended next source: `return_preserving_hybrid_target_bc_replay_v1`.
- Recommended next action: `inspect_selected_hybrid_failure_and_repair_participation_weight_or_target_before_retention_training`.
- Primary blocker: `walkforward_relative_return_gap`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
