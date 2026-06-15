# Model Retention Frontier

- Rows scanned: `173`.
- Completed handoffs: `136`.
- Recommended next source: `release16_holdout_slice_repair_candidate_screen_v1`.
- Recommended next action: `train_state_conditioned_action_head_or_aux_state_loss_then_repeat_holdout_slice_screen`.
- Primary blocker: `bc_candidate_or_slice_gate_failed`.
- Retention gate: MaxDD `< 20.0`, WF relative return `>= 30.0` pp, and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
