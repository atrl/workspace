# Model Retention Frontier

- Rows scanned: `158`.
- Completed handoffs: `122`.
- Recommended next source: `release16_nonconvex_checkpoint_bc_official_replay_v1`.
- Recommended next action: `repair_sparse_slice_fit_or_state_head_feature_before_ppo`.
- Primary blocker: `bc_candidate_or_slice_gate_failed`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
