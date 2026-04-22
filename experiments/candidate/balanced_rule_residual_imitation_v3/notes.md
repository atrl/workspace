# balanced_rule_residual_imitation_v3

## Motivation

v2 achieved good test_2022 control (MaxDD 30.2%, return -21.9% vs rulebase -48.2%) but failed on walk-forward 2023-2024:
- test_2023_2024: 34.0% / 0.63 / 29.4% vs rulebase 112.5% / 0.96 / 41.6%
- Model is too conservative in bull markets, missing major rallies

Additionally, v2 training degenerated after iter 0 (52.1% → 39.7% on validation).

## Hypothesis

Reducing max_iters to 40 + increasing participation signals + reducing imitation bonus will:
1. Capture more bull market upside via trend_alignment and opportunity_cost
2. Prevent training degeneration by stopping earlier
3. Maintain test_2022 control while improving walk-forward returns

## Mechanism Delta (vs imitation_v2)

| Parameter | v2 | v3 |
|-----------|-----|-----|
| max_iters | 80 | 40 |
| alpha_scale | 3.0 | 3.5 |
| episode_drawdown_penalty | 3.5 | 3.0 |
| drawdown_penalty | 0.30 | 0.25 |
| trend_alignment_bonus | 0.25 | 0.35 |
| opportunity_cost_penalty | 0.05 | 0.10 |
| rule_imitation_bonus | 0.10 | 0.05 |
| rule_decision_bonus | 0.15 | 0.08 |

## Acceptance Criteria

- test_2022 MaxDD < 30%
- test_2023_2024 return > 60% (vs v2's 34.0%, closer to rulebase 112.5%)
- test_2023_2024 Sharpe > 0.80
- test_proxy Sharpe > 0.85

## Failure Mode

- If too loose on drawdown, test_2022 MaxDD may exceed 30%
- If participation signals too strong, model may overtrade
