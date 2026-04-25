# Expectations: risk_budget_usage_v2

## What we expect to improve

1. **Proxy Sharpe**: from 0.942 → 0.95+ via mode-conditional reward guiding
   less conservative usage policies.
2. **2022 stress resilience**: maintained or slightly better via dynamic band
   compression in high-vol periods.
3. **2023-2024 restart participation**: maintained via crash_rebuild_bonus
   encouraging bottom-fishing.

## What we accept may get worse

1. **Test_2020 return**: could drop slightly if dynamic bands compress too much
   during the COVID crash recovery.
2. **Turnover**: mode-conditional rewards may increase action frequency.

## Failure modes to watch

1. **Overfitting to mode semantics**: if the reward shaping is too strong, the
   actor may ignore market signals and purely chase mode bonuses.
2. **Band oscillation**: dynamic scaling could cause band edges to oscillate
   daily, leading to increased turnover.
3. **Vol lag**: realized_vol_10 is a 10-day lookback; sudden vol spikes may
   not be captured fast enough.

## Abort criteria

- test_2022 MaxDD > 25%
- test_proxy Sharpe < 0.90
- test_2023_2024 return < 60%
