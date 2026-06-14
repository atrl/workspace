# Model Retention Frontier

- Rows scanned: `134`.
- Completed handoffs: `110`.
- Recommended next source: `release16_rulebase_reversal_checkpoint_metric_replay_v1`.
- Recommended next action: `diagnose_remaining_rulebase_reversal_failed_folds_before_full_retention_training`.
- Primary blocker: `walkforward_relative_return_gap`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
