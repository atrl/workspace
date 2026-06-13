# Model Retention Frontier

- Rows scanned: `94`.
- Completed handoffs: `64`.
- Recommended next source: `cap10_metric_replay_handoff_audit_v1`.
- Recommended next action: `inspect_failed_fold_drawdown_or_reward_drift_before_full_retention_training`.
- Primary blocker: `walkforward_relative_return_gap`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
