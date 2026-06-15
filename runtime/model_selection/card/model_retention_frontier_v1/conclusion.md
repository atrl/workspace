# Model Retention Frontier

- Rows scanned: `174`.
- Completed handoffs: `137`.
- Recommended next source: `release16_state_aux_loss_training_capability_v1`.
- Recommended next action: `run_aux_state_loss_bc_holdout_slice_screen_with_trend_preservation_floor_before_checkpoint_screen`.
- Primary blocker: `training_contract_ready`.
- Retention gate: MaxDD `< 20.0`, WF relative return `>= 30.0` pp, and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
