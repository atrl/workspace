# Strict Retention Training Handoff

- decision: `train_bull_trend_participation_with_drawdown_drift_stop_no_promotion`
- promotion: `no promotion`
- actor_retention_allowed: `False`
- ppo_allowed_next: `False`
- reference_baseline: `soxl_backtest_policy_reimplementation_v1`

## Objective Terms

- `soxlbacktest_reference_alignment`
- `walkforward_relative_return_gap_penalty`
- `checkpoint_objective_loss`
- `wf2020_trend_participation_floor`
- `drawdown_drift_penalty`

## Fold Blockers

- wf_2020: `trend_underparticipation`, relative=-285.449375pp, MaxDD=19.417943%

## Stop Conditions

- `do_not_promote_missing_walkforward_windows`
- `do_not_promote_worst_wf_maxdd_ge_20`
- `do_not_promote_ytd_return_below_best_alignment`
- `do_not_promote_without_soxlbacktest_reference_alignment`
- `do_not_continue_same_ppo_when_maxdd_drift_exceeds_2pp`
