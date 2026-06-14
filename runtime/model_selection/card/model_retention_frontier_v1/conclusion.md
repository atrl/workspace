# Model Retention Frontier

- Rows scanned: `103`.
- Completed handoffs: `69`.
- Recommended next source: `wf2020_risk_cap_reference_target_screen_v1`.
- Recommended next action: `run_file_backed_wf2020_riskcap_full_bc_replay_before_any_retention_then_design_return_preserving_hybrid_target_if_ytd_gate_fails`.
- Primary blocker: `training_contract_ready`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
