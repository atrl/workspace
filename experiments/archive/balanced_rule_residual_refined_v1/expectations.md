# Expectations: balanced_rule_residual_refined_v1

## What must improve

1. **proxy MaxDD** must be < 58.5% (simplified_v1) and ideally < 54.8% (v19)
2. **test_2022 MaxDD** must be < 34% (simplified_v1) and ideally < 31% (v19)
3. **cash holding** should be more common in high-risk regimes

## What we watch for

- Validation score trajectory (staged selection at iter 20)
- Episode return variance (should be lower with cleaner incentives)
- Trade count (may decrease due to removed activity_bonus)
- Proxy Sharpe (target > 0.9)

## Success criteria

| metric | threshold |
|--------|-----------|
| proxy return | > 500% |
| proxy sharpe | > 0.85 |
| proxy maxdd | < 55% |
| test_2022 maxdd | < 33% |
| test_2022 return | > -25% |
| test_2023_24 return | > 40% |

## Next action if success

1. Apply conditional drawdown budget overlay (regime_risk >= 0.6)
2. Run multi-asset joint training with this reward design
3. Compare with mode_v1 structural approach

## Next action if failure

1. If DD improves but return collapses: increase opportunity_cost_penalty slightly
2. If DD does not improve: issue is structural — pivot to state/execution changes
3. If cash is held too much: add minimal trend-alignment bonus back
