# balanced_rule_residual_imitation_v6

## Motivation

v4 achieved the best walk-forward returns (51.8% test_2023_2024) with MaxDD just 1-1.3pp over 30%:
- test_2022 MaxDD: 31.3% (target: < 30%)
- test_2023_2024 MaxDD: 31.1% (target: < 30%)

v5 showed that tightening episodic penalties further is counterproductive (degeneration).

## Hypothesis

Lowering hard_floor from 0.20 to 0.10 will allow the model to reduce exposure more aggressively in bear markets, pushing MaxDD below 30% without affecting the participation signals that drive bull market returns.

## Mechanism Delta (vs imitation_v4)

| Parameter | v4 | v6 |
|-----------|-----|-----|
| hard_floor | 0.20 | 0.10 |
| Everything else | v4 | same |

## Acceptance Criteria

- test_2022 MaxDD < 30%
- test_2023_2024 MaxDD < 30%
- test_2023_2024 return > 45%
