# Model Retention Frontier

- Rows scanned: `172`.
- Completed handoffs: `135`.
- Recommended next source: `release16_joint_target_holdout_slice_repair_v1`.
- Recommended next action: `repair_state_conditioned_action_head_or_aux_state_loss_before_reward_checkpoint_screen`.
- Primary blocker: `bc_candidate_or_slice_gate_failed`.
- Retention gate: MaxDD `< 20.0`, WF relative return `>= 30.0` pp, and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
