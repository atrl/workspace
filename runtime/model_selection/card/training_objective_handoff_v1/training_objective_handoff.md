# Training Objective Handoff

## Decision

- decision: `train_state_conditioned_multi_objective`
- promotion: `blocked`
- selected_diagnostic_candidate: `recovery_reentry_conditional_vol_target_v1`

## Objective Terms

- `walkforward_relative_return_gap_penalty`
- `walkforward_drawdown_gap_penalty`
- `high_vol_rebound_participation_floor`
- `downtrend_drawdown_state_cap`

## Risk Gap

- max_dd_gap=11.14pp, relative_gap=0.0pp, worst_window=`wf_2020`

## Primary Underparticipation Cluster

- cluster=`C003`, start=2026-02-04, end=2026-05-21, mean_excess=-167.30pp

## Stop Mechanisms

- `path_level_threshold_router_search`
