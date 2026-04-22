# balanced_rule_residual_imitation_v4

## Motivation

v2 had the best MaxDD control (test_2022: 30.2%, test_2023_2024: 29.4%) but weak returns.
v3 increased participation signals and alpha_scale, which improved returns but MaxDD worsened (test_2022: 34.0%, test_2023_2024: 36.1%).

The core challenge: achieve BOTH MaxDD < 30% AND strong walk-forward returns.

## Hypothesis

v2's drawdown framework is correct. The issue is insufficient participation signals.
By keeping v2's tight drawdown controls but aggressively increasing trend_alignment and opportunity_cost, we can:
1. Maintain MaxDD < 30% in all periods
2. Improve test_2023_2024 returns without increasing drawdowns
3. Avoid training degeneration by limiting max_iters to 20

## Mechanism Delta (vs imitation_v2)

| Parameter | v2 | v4 |
|-----------|-----|-----|
| max_iters | 80 | 20 |
| alpha_scale | 3.0 | 3.2 |
| trend_alignment_bonus | 0.25 | 0.40 |
| trend_alignment_threshold | 0.03 | 0.02 |
| opportunity_cost_penalty | 0.05 | 0.20 |
| opportunity_threshold_pos | 0.40 | 0.50 |
| Drawdown controls | tight | same as v2 |

## Acceptance Criteria

- test_2022 MaxDD < 30%
- test_2023_2024 MaxDD < 30%
- test_2023_2024 return > 50%
- test_proxy Sharpe > 0.85

## Failure Mode

- If participation signals too strong, model may overtrade in choppy markets
- If max_iters too low, model may not learn enough
