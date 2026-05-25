# Training Objective Handoff

## Decision

- decision: `train_state_conditioned_multi_objective`
- promotion: `blocked`
- selected_diagnostic_candidate: `low_floor_rebound_dd10`

## Objective Terms

- `walkforward_relative_return_gap_penalty`
- `walkforward_drawdown_gap_penalty`
- `high_vol_rebound_participation_floor`
- `downtrend_drawdown_state_cap`

## Risk Gap

- max_dd_gap=7.02pp, relative_gap=173.58pp, worst_window=`cap10_release_0.25__wf_2023_2024`

## Primary Underparticipation Cluster

- cluster=`C003`, start=2023-10-23, end=2024-03-07, mean_excess=-112.97pp

## Stop Mechanisms

- `path_level_threshold_router_search`
