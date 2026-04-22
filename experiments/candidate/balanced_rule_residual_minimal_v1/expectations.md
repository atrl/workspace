# Expectations: balanced_rule_residual_minimal_v1

## Baseline

- Direct comparison: `balanced_rule_residual_v16` (GRU, window=5, penalty=2.0)
  - 632.4% proxy return / 0.91 sharpe / 50.1% MaxDD (seed=20260420)
  - 432.7% proxy return / 0.81 sharpe / 50.8% MaxDD (seed=42)
- Canonical baseline: `balanced_rule_residual_v19`
  - 735.1% proxy return / 0.90 sharpe / 54.8% MaxDD

## Expected Behavior

1. Training should be more stable (fewer gradient conflicts)
2. Valid MaxDD should be lower than v16 because regime_budget provides structural constraint
3. Return may be lower than v16 because we removed trend_alignment and opportunity_cost incentives
4. Trades should be fewer than v16 (~334) because no activity_bonus forces trading
5. Stress average position should be lower because regime_budget caps at 0.30 in high-risk

## Red Flags

- valid best iter=0 (no learning)
- proxy return < 400% (permanent cash)
- proxy MaxDD > 55% (budget not working)
- test_2022 MaxDD > 38% (crisis behavior worse than v16)
- trades < 100 (complete inactivity)

## Success Indicators

- valid MaxDD < 28% (v16 was 28.9%)
- proxy MaxDD < 48%
- test_2022 MaxDD < 33%
- proxy return > 500%
- trades between 200-350
