# Expectations for balanced_rule_residual_v15

## Must-haves (hard criteria)

- [ ] Training completes without shape/runtime errors
- [ ] Summary artifact persists to `artifacts/models/`
- [ ] Fixed-split promotion gate evaluation runs

## Nice-to-haves (soft criteria)

- [ ] MaxDD < 50% (better than MLP baseline)
- [ ] Return > 550% (better than v12/v13 conservatism)
- [ ] test_2022 MaxDD < 35% (promotion gate)

## Drop conditions

- [ ] MaxDD > 56% (worse than v14, penalty has no effect at 1.0)
- [ ] Return < 500% AND MaxDD > 52% (worse than v12 on both axes)
