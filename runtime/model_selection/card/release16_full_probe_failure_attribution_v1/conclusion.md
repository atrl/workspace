# release16_full_probe_failure_attribution_v1 conclusion

Decision: `release16_full_probe_failure_attribution_no_actor_retention`.

The latest release16 full probe remains non-retainable. The converted SOXLBacktest qlib reference is still accepted as the YTD baseline: return 132.1%, MaxDD 13.6%, legacy delta 0pp. Release16 YTD return is 38.6% with relative -288.4pp and avg position 0.085, which fails both SOXLBacktest reference and current-best return floors despite YTD MaxDD staying below 20%.

## Poor-window clusters

- `C006` 2026-02-26..2026-06-09: type `underparticipation`, mean excess -197.4pp, max MaxDD 7.7%, mean pos 0.007.
- `C005` 2025-04-08..2025-07-21: type `underparticipation`, mean excess -205.9pp, max MaxDD 2.0%, mean pos 0.347.
- `C001` 2020-03-16..2020-08-13: type `underparticipation`, mean excess -126.2pp, max MaxDD 4.3%, mean pos 0.285.

## Feature-state consensus

- `trend_quality`: mentions 15, mean |delta_z| 0.93; examples anchor_gap_velocity_10;trend_gap_21;anchor_gap_180;mom_21;below_anchor_ratio_63
- `volatility_regime`: mentions 12, mean |delta_z| 1.43; examples tracking_error_vol_21;vix_level;realized_vol_63;vol_coef_21;vol_of_vol_21
- `other`: mentions 9, mean |delta_z| 0.78; examples gain_streak_10;expectancy_proxy_21;sharpe_proxy_21;rsi_14;rsi_6
- `drawdown_path`: mentions 6, mean |delta_z| 0.72; examples drawdown_streak_63;days_since_high_63;drawdown_63;cum_drop_30;drawdown_21
- `path_decay_or_tracking`: mentions 4, mean |delta_z| 1.73; examples tracking_error_21;decay_63;jump_variation_5;tracking_error_5

## Handoff

`derive_training_objective_handoff()` returns `train_state_conditioned_multi_objective` with objective terms: walkforward_relative_return_gap_penalty, walkforward_drawdown_gap_penalty, high_vol_rebound_participation_floor, downtrend_drawdown_state_cap. This means the next admissible work is a state-conditioned multi-objective target/checkpoint screen, not another unchanged PPO run and not actor retention.

Next action: build or reuse a fold-local objective/target screen that jointly penalizes walk-forward relative-return gaps and drawdown gaps, with explicit high-vol/rebound participation and downtrend drawdown-state control. Any follow-up training branch must still pass fixed split, `wf_2020`, `wf_2022`, `wf_2023_2024`, YTD SOXLBacktest/current-best gates, model replay export, and dashboard runnable-source checks before retention.
