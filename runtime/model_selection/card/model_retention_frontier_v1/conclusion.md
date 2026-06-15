# Model Retention Frontier

- Rows scanned: `169`.
- Completed handoffs: `132`.
- Recommended next source: `release16_drawdown_drift_short_training_screen_v1`.
- Recommended next action: `revise_reward_patch_or_checkpoint_selection_before_full_training`.
- Primary blocker: `short_training_gate_failed`.
- Retention gate: MaxDD `< 20.0`, WF relative return `>= 30.0` pp, and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
