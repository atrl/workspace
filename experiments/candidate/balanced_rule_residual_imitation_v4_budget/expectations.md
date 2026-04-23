# balanced_rule_residual_imitation_v4_budget Expectations

## Role

Execution-layer candidate over `balanced_rule_residual_imitation_v4`, not a new reward or architecture experiment.

## Expected Behavior

- Same actor architecture and reward intent as `balanced_rule_residual_imitation_v4`.
- Position budget remains inactive below `regime_risk=0.60`.
- Once high-risk regime is active, allowed position depends on consumed drawdown budget:
  - under 30% consumed: no extra cap
  - 30-60% consumed: max position 0.70
  - 60-90% consumed: max position 0.45
  - 90%+ consumed: max position 0.15

## Evaluation Priority

1. Fixed split: `test_2022`, `test_2023_2024`, `test_proxy`.
2. Walk-forward only if fixed-split evidence beats or clearly complements v4.
3. Durable comparison row in `docs/experiment-landscape-review.md` after evaluation.
