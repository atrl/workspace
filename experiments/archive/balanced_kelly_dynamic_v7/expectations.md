# Expectations: balanced_kelly_dynamic_v7

## What Must Improve

1. **Position dynamics**: test_2022 pos_std must exceed 0.05. Model must show
   at least some response to Kelly rebound signals.

2. **Walk-forward return**: test_2023_2024 Return must exceed v6's 45.61%.
   Target: > 60%.

## Accepted Trade-off

- test_2022 MaxDD may expand to 35-38% due to weaker episodic constraint (1.5 vs 3.5).
- But if Kelly bonus works, model should also capture some upside in 2022 rebounds.

## Go/No-Go Criteria

- **GO**: pos_std > 0.05 AND test_2023_2024 Return > 55%
- **NO-GO**: pos_std < 0.03 (still fixed) OR test_2022 MaxDD > 40%
