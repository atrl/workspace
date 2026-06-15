# Training Objective Handoff

## Decision

- decision: `train_nonconvex_checkpoint_objective_with_sparse_slice_guard`
- promotion: `blocked`
- selected_diagnostic_candidate: `release16_nonconvex_checkpoint_objective_readiness_v1::per_fold_teacher_mix`

## Objective Terms

- `official_replay_relative_floor_loss`
- `official_replay_drawdown_drift_stop_loss`
- `state_head_sparse_slice_fit_guard`
- `fold_local_teacher_alignment_loss`

## Risk Gap

- max_dd_gap=1.65pp, relative_gap=8.67pp, worst_window=`wf_2023_2024`

## Primary Underparticipation Cluster

- cluster=`wf_2020_relative_floor_and_drawdown_drift`, start=2020-01-02, end=2020-12-30, mean_excess=-23.42pp

## Stop Mechanisms

- `path_level_threshold_router_search`
- `actor_retention_before_official_three_fold_replay`
