# Expectations: aggressive

## Acceptance Criteria

- proxy return >= balanced baseline
- proxy Sharpe >= 0.75
- test_2022 MaxDD <= 35%

## Expected Trade-off

- Higher return_scale → higher return, worse drawdown
- Weaker penalties → more volatile behavior
- Stronger activity bonus → higher turnover

## Failure Modes

- Drawdown exceeds 50% → penalties too weak
- Turnover too high → activity_bonus excessive
- Seed instability → aggressive config sensitive to initialization

## Comparison-Table Row Key

`aggressive`
