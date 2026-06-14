# Model Retention Frontier

- Rows scanned: `125`.
- Completed handoffs: `101`.
- Recommended next source: `rebound_state_factor_target_repair_screen_v1`.
- Recommended next action: `run_release16_cb99_foldlocal_checkpoint_or_ppo_screen_before_actor_retention`.
- Primary blocker: `target_revision_ready`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
