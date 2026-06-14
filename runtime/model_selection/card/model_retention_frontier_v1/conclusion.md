# Model Retention Frontier

- Rows scanned: `100`.
- Completed handoffs: `66`.
- Recommended next source: `bull_reentry_three_fold_checkpoint_metric_replay_v1`.
- Recommended next action: `inspect_failed_fold_drawdown_or_reward_drift_before_full_retention_training`.
- Primary blocker: `walkforward_relative_return_gap`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
