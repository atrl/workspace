# Expectations: balanced_kelly_execution_v8

## What Must Improve

1. **test_2022 position dynamics**: Average position during Kelly > 0.2 periods
   (Jul/Aug, Nov, Dec 2022) must be visibly higher than v4's ~0.26.

2. **test_2023_2024 return**: Must exceed v4's 51.76%.
   Mechanical floor-raising in early 2023 (Kelly > 0.3 in Jan-Feb) should
   force higher exposure during the bull restart.

## Accepted Trade-off

- test_2022 MaxDD may increase slightly if mechanical floor-raising happens
  just before a dip. Vol attenuation (realized_vol > 1.0) limits this.

## Go/No-Go Criteria

- **GO**: test_2023_2024 Return > 65% AND test_2022 MaxDD < 36%
- **NO-GO**: test_2023_2024 Return < 50% (no improvement over v4) OR
  test_2022 MaxDD > 40%
