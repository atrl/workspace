# Model Retention Frontier

- Rows scanned: `161`.
- Completed handoffs: `125`.
- Recommended next source: `release16_joint_drawdown_aux_state_head_bc_holdout_replay_v1`.
- Recommended next action: `repair_action_head_sampling_or_aux_state_loss_before_official_replay`.
- Primary blocker: `bc_candidate_or_slice_gate_failed`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
