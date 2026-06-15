# release16_wf2020_foldlocal_trend_target_v1

Decision: `release16_wf2020_foldlocal_trend_target_ready_for_bc_screen_no_promotion`.

Selected candidate: `wf2020_lagged_trend_participation_floor`.

Target readiness: `True`. Promotion allowed: `False`. Actor retention allowed: `False`.

Key evidence:

- wf2020 trend target rows: `357`.
- total trend target rows: `631`.
- actual policy-label samples: `3417`.
- actual unique actions: `5`.
- dominant action share: `0.43985952589991223`.
- eval overlap rows: `0`.

Risk note: the recipe parameters came from an OOS wf2020 lagged counterfactual. This card only authorizes a downstream BC/checkpoint official replay screen; it does not retain or promote an actor. The counterfactual `max_equity_drawdown=0.2` guard is kept for downstream replay and is not transferred into row labels.

Next action: `run_release16_wf2020_trend_target_bc_checkpoint_official_three_fold_replay_with_drawdown_drift_stop`.
