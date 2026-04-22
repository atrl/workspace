# Expectations: balanced_rule_residual_moderate_v1

## Baseline

- Direct comparison: `balanced_rule_residual_v16` (seed=20260420)
  - 632.4% proxy return / 0.91 sharpe / 50.1% MaxDD / 334 trades
- Alternative comparison: `balanced_rule_residual_refined_v3` (seed=12345)
  - 637.8% proxy return / 0.89 sharpe / 54.9% MaxDD

## Expected Behavior

1. Training should be stable (no degradation from iter 0 like minimal_v1)
2. Valid MaxDD should be ~28-30% (similar to v16)
3. Return should be between v16 and refined_v3 (580-650% range)
4. Trades should be ~300-350
5. Step-level penalties are weaker, so stress behavior may be slightly more aggressive

## Red Flags

- valid best iter=0 (no learning, like minimal_v1)
- proxy return < 500% (participation signals insufficient)
- proxy MaxDD > 55% (step penalties too weak)
- test_2022 MaxDD > 38%

## Success Indicators

- valid MaxDD < 30%
- proxy MaxDD < 50%
- proxy return > 550%
- test_2022 MaxDD < 33%
- trades between 280-380
