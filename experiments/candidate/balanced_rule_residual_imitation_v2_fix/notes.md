# balanced_rule_residual_imitation_v2_fix

## Problem Diagnosed

v2 achieved test_2022 MaxDD=30.2% (close to target) but diagnosis revealed the model spent most of 2022 at 0% position, then suddenly went 100% right before the October crash:

```
2022-10-07: model_pos=1.00, market=-17.69% → single day -17.7%
2022-10-10: model_pos=1.00, market=-10.10% → single day -10.1%
```

Root cause: `panic_buy_bonus` encourages the model to "catch falling knives" in extreme volatility. When cum_drop exceeds -0.15 and regime_risk > 0.60, the model gets a bonus for buying — but in Oct 2022, there was no bottom.

## Fix Applied

1. **Remove panic_buy_bonus** (`panic_buy_bonus: 0.0`)
2. **Add bottomless_pit_penalty** (`bottomless_pit_penalty: 0.50`)
   - When cum_drop_30 < -0.25 and model increases position: -0.25 penalty
   - When cum_drop_30 < -0.35: additional -0.25
   - When realized_vol > 1.0: -0.15 additional
   - When realized_vol > 1.5: another -0.15
   - When regime_risk > 0.75: another -0.10
   - Maximum penalty per step: 0.50 * 0.90 = 0.45

## Hypothesis

Removing panic_buy + adding bottomless pit protection will:
1. Prevent the model from going 100% before market crashes
2. Keep test_2022 MaxDD below 30%
3. Maintain or improve test_2023_2024 returns (since 2023-2024 is a bull market, not a falling knife scenario)

## Acceptance Criteria

- test_2022 MaxDD < 30%
- test_2023_2024 MaxDD < 30%
- test_2023_2024 return > 30%
