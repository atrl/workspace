# Expectations for balanced_rule_residual_v16

## Must-haves

- [ ] Training completes without errors
- [ ] Summary artifact persists
- [ ] GRU tensor shapes handle observation_window=5 correctly

## Nice-to-haves

- [ ] MaxDD < 50% (match v12/v13 drawdown control)
- [ ] Return > 520% (beat v12/v13, match MLP baseline)
- [ ] test_2022 MaxDD < 35% (promotion gate)

## Drop conditions

- [ ] Runtime errors from GRU shape mismatch
- [ ] MaxDD > 55% (worse than MLP baseline)
- [ ] Return < 450% AND MaxDD > 52% (worse than v12 on both axes)
