# Model Retention Frontier

- Rows scanned: `93`.
- Completed handoffs: `50`.
- Recommended next source: `cap10_bc_sequence_risk_state_diagnostic_v1`.
- Recommended next action: `add_sequence_risk_state_cap_to_joint_contract_then_screen_bc_and_checkpoint_drift`.
- Primary blocker: `missing_walkforward_evidence`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
