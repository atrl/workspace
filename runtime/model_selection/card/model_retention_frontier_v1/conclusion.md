# Model Retention Frontier

- Rows scanned: `92`.
- Completed handoffs: `47`.
- Recommended next source: `state_objective_budget_usage_bc_official_replay_v1`.
- Recommended next action: `run_fold_local_ppo_only_if_ytd_and_wf_gap_is_not_structural_else_revise_targets`.
- Primary blocker: `joint_drawdown_return_gap`.
- Retention gate: MaxDD `< 20.0` and YTD return floor `213.143462` versus SOXLBacktest reference `132.077978`.

This card ranks model-selection evidence for the next training run. It does not promote or retain actors.
