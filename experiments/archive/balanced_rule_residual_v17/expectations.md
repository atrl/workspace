# Expectations for balanced_rule_residual_v17

## Must-haves

- [ ] Training completes without tensor shape errors
- [ ] GRU handles observation_window=10 correctly

## Nice-to-haves

- [ ] Return > 600% (match v16)
- [ ] MaxDD < 52% (comparable to v16)
- [ ] Sharpe > 0.88

## Drop conditions

- [ ] Runtime errors from GRU window mismatch
- [ ] Return < 500% AND MaxDD > 55%
