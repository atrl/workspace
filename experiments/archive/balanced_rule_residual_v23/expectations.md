# Expectations for balanced_rule_residual_v23

## Hypothesis

`balanced_rule_residual_v19` already solves the broad stress-defense problem but
still lags on stress-release and restart windows. A narrow transition-state
feature pack should improve selective re-entry timing without repeating the old
failure mode where transition features simply raise full-window `2022`
exposure.

## Deltas vs v19

- Feature schema grows from 59 to 65 columns (+6 transition-state features).
- Reward, position cap, model, execution, and sampling remain identical to
  `v19`.
- Obs dim grows from 315 to 345 for `observation_window=5`.
- Existing `v19` actors are not backward-compatible; `v23` is a fresh run.

## Must-haves

- [ ] Training completes under `SOXL_USE_TRANSITION_V23=1`.
- [ ] `feature_schema.signature` differs from `v19`.
- [ ] `test_2022_max_dd <= 35%`.
- [ ] `test_2022_avg_pos <= 0.35`.
- [ ] `test_proxy_sharpe >= 0.80`.

## Nice-to-haves

- [ ] `test_2020` return >= `v19` (+3pp), or maxDD improves by at least 1pp.
- [ ] At least two of `2022-07`, `2022-11`, `2023-01` improve versus `v19`.
- [ ] `test_2023_2024` return is flat to slightly higher without worsening
  drawdown materially.

## Kill criteria

- If `test_2022_avg_pos` rises by more than `+0.02` vs `v19`, revert.
- If `test_2022_max_dd` worsens by more than `+2pp` vs `v19`, revert.
- If local rebound windows improve but full `2022` discipline worsens, revert.

## Comparison anchor

- `balanced_rule_residual_v19` (seed=12345, 59-feature)
- Slice report target:
  `runtime/model/reviews/balanced_rule_residual_v23_vs_v19.md`
