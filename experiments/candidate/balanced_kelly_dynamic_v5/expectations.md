# Expectations: balanced_kelly_dynamic_v5

## What Must Improve

1. **Dynamic position in 2022**: Average position should vary significantly
   (std > 0.10) instead of being fixed at ~0.20. Position should spike during
   recoveries and drop during crash legs.

2. **Walk-forward 2023-2024 return**: Must exceed v4's 51.8%. Target: > 80%.
   Kelly features should enable the model to capture more upside in bull markets.

3. **test_2022 MaxDD**: Must remain < 33%. The Kelly tracking has vol
   attenuation to prevent excessive exposure in extreme volatility.

## What Should Not Regress

1. **test_2022 MaxDD**: Must not exceed 33% (promotion gate is 30%, but we
   accept slight relaxation for significant walk-forward improvement).

2. **Turnover**: Should not exceed 2.0x annually (v4 level).

3. **test_2020 performance**: Should remain > 100%.

## Metrics to Watch

- `avg_pos` and `pos_std` in test_2022 (diagnostic for fixed-position collapse)
- `kelly_position_21` correlation with actual position (feature utilization)
- Episode-level return distribution (should show wider spread, not just
  clustered near cash-holding)

## Go/No-Go Criteria

- **GO**: test_2022 MaxDD < 33% AND test_2023_2024 Return > 70%
- **NO-GO**: test_2022 MaxDD > 35% OR model still shows fixed position
  (std < 0.05) OR test_2023_2024 Return < 50%
