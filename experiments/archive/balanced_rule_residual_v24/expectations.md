# Expectations for balanced_rule_residual_v24

## Motivation

`balanced_rule_residual_v19` is the retained high-return anchor, but the repo's
older decision-frequency experiments suggest that changing the decision
timescale is one of the few structural levers that can lower drawdown without
falling back into reward tuning or overlay drift.

## External Reference

- Sutton, Precup, Singh (1999), `Between MDPs and Semi-MDPs`
- Read: temporally abstract actions are a principled way to reduce myopic
  primitive-action churn

## Hypothesis

Using `decision_frequency=2` plus a compressed residual action set on top of the
retained `v19` family will:

- reduce trade count
- improve `test_2022` drawdown discipline
- preserve more return than the older `decision_frequency=3` compression loops

## Result

- partial hit on `test_2022` control:
  - `max_dd 31.07 -> 30.67`
  - `avg_pos 0.261 -> 0.257`
- failed on the main intended mechanism:
  - trade count expanded sharply instead of compressing
- not promotable as a replacement for `v19`

## Must-haves

- [x] Training completes
- [x] `test_2022 max_dd` improves versus `balanced_rule_residual_v19`
- [ ] `trade_count` improves versus `balanced_rule_residual_v19`

## Nice-to-haves

- [x] `test_2023_2024 return >= 35%`
- [x] `test_proxy_sharpe >= 0.80`
- [x] proxy return remains above `500%`

## Failure Mode

- the model becomes broadly too defensive, like other compression / transition
  loops, and loses recovery return without enough 2022 benefit

## Analysis Target

- isolate whether medium-strength temporal abstraction is a real improvement on
  the `v19` family or whether decision-timescale changes only work on the older
  floor40 branch
- current answer:
  - on the retained `v19` family, this temporal-compression variant lowers
    2022 exposure a little but does not reduce realized churn
