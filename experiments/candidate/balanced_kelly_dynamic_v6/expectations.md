# Expectations: balanced_kelly_dynamic_v6

## What Must Improve

1. **Position dynamics**: test_2022 position std must exceed 0.10 (v5 was ~0.00).
   Model must show visible response to Kelly rebound signals in Jul/Aug and Nov.

2. **Walk-forward return**: test_2023_2024 Return must exceed v5's 45.61%.
   Target: > 70%.

## What Should Not Regress

1. **test_2022 MaxDD**: Must remain < 35%. With episode_drawdown reduced to 2.5,
   there is risk of MaxDD expansion, but Kelly vol attenuation should help.

2. **test_2020 performance**: Should remain > 100%.

## Go/No-Go Criteria

- **GO**: test_2022 pos_std > 0.08 AND test_2023_2024 Return > 60%
- **NO-GO**: test_2022 pos_std < 0.05 (still fixed) OR test_2022 MaxDD > 38%
