# retention_candidate_overlay_screen_v1

- Decision: `overlay_target_screen_no_actor_promotion`.
- Source profiles screened: `state_conditioned_drawdown_rebound_reward_train_v1, drawdown_rebound_reward_train_v1, mode_conditioned_rebound_reward_train_v2`.
- Source export candidates: `3`; position replay overlays: `207`; strict gate pass count: `0`.
- Gate: WF windows `wf_2020, wf_2022, wf_2023_2024`, worst WF MaxDD `< 20.00%`, YTD return `>= 132.077978%` and `>= 213.143462%`.
- Best official source export: `drawdown_rebound_reward_train_v1__source_export_nav`.
- Best official source metrics: YTD `172.83%`, YTD MaxDD `27.31%`, worst WF MaxDD `20.93%`, failures `worst_wf_max_drawdown_ge_20;ytd_return_below_current_best_alignment`.
- Best position-replay diagnostic: `state_conditioned_drawdown_rebound_reward_train_v1__scale1.40_dd63_m0.30_s0.10` with YTD `0.50%` and worst WF MaxDD `18.68%`.
- Position replay is diagnostic only: exported `position` plus SOXL price does not reproduce `strategy_nav`, so it cannot be used for retention or promotion scoring.
- No existing complete-WF export or simple position-path overlay satisfies the final retention gate. This blocks actor retention and confirms the next useful work is actual cap10/joint-contract BC/checkpoint training with explicit WF coverage and drift stop, not a dashboard/registry workaround.

## Sidecars
- `runtime/model_selection/card/retention_candidate_overlay_screen_v1/source_daily_manifest.csv`
- `runtime/model_selection/card/retention_candidate_overlay_screen_v1/overlay_grid_summary.csv`
- `runtime/model_selection/card/retention_candidate_overlay_screen_v1/overlay_window_metrics.csv`
- `runtime/model_selection/card/retention_candidate_overlay_screen_v1/summary.json`
