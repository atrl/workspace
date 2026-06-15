# Model Retention Frontier

- Rows scanned: `166`.
- Completed handoffs: `129`.
- Recommended next source: `release16_wf2020_trend_target_bc_official_replay_v1`.
- Recommended next action: `repair_trend_slice_sampling_or_state_loss_before_official_replay`.
- Primary blocker: `bc_candidate_or_slice_gate_failed`.
- Retention gate: MaxDD `< 20.0`, WF relative return `>= 30.0` pp, and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
