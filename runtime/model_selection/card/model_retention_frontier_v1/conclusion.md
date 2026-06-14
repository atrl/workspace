# Model Retention Frontier

- Rows scanned: `109`.
- Completed handoffs: `83`.
- Recommended next source: `short_volshock_drawdown_drift_feature_audit_v1`.
- Recommended next action: `screen drawdown-drift reward/checkpoint penalty with trend-preservation term before PPO`.
- Primary blocker: `missing_walkforward_evidence`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
