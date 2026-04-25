# v19_small Training Complete: Worst Bear-Market Performance in Family

Date: 2026-04-25

## Results

`balanced_rule_residual_v19_small` (smaller network variant of v19) completed training.

| Metric | Valid | test_2020 | test_2022 | test_2023_2024 |
|--------|-------|-----------|-----------|----------------|
| Return | +62.4% | +96.9% | **-31.6%** | +46.8% |
| Sharpe | 1.57 | 2.04 | **-1.11** | 0.67 |
| MaxDD | 30.7% | 26.6% | 38.3% | 41.9% |
| AvgPos | 0.66 | 0.53 | 0.31 | 0.60 |
| StressDays | — | 48 | 129 | 64 |

## Comparison with Family

| Profile | test_2022 Sharpe | test_2022 MaxDD | test_2023_2024 Sharpe | test_2023_2024 MaxDD |
|---------|-----------------|-----------------|----------------------|---------------------|
| balanced (canonical) | -0.91 | 33.2% | — | — |
| v19 | -0.80 | 35.2% | — | — |
| v22 | -0.76 | 30.1% | 0.72 | 35.7% |
| **v19_small** | **-1.11** | **38.3%** | **0.67** | **41.9%** |

## Key Findings

1. **Worst bear-market performance in the family** — test_2022 Sharpe -1.11 is significantly worse than v19 (-0.80), v22 (-0.76), and balanced (-0.91)
2. **test_2023_2024 MaxDD 41.9%** — very high drawdown in the recovery window, worse than v22 (35.7%)
3. **Proxy Sharpe 0.84** — passes the 0.75 threshold, but fixed-split metrics are misleading (as proven repeatedly)
4. **Smaller network does NOT help** — v19_small uses a reduced network but performs worse on all stress metrics

## Conclusion

Network size reduction is NOT the solution to bear-market generalization. v19_small is the worst-performing residual family member on bear markets.

## Decision

- `balanced_rule_residual_v19_small`: **dropped**
- `balanced_rule_residual_v19_tiny` (baseline: v19_small): should not be trained — predicted to fail similarly

## Pattern Confirmation

Four consecutive confirmations of family-level bear-market failure:
1. balanced (canonical): 2022 Sharpe -0.91
2. v19: 2022 Sharpe -0.80
3. v22: 2022 Sharpe -0.76
4. v19_small: 2022 Sharpe -1.11

The problem is NOT network size, reward coefficients, or residual architecture. The problem is **training data lacks prolonged bear markets**.
