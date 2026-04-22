# balanced_rule_residual_imitation_v2

## Motivation

Imitation v1 achieved 812.3% proxy return but failed on critical OOS metrics:
- test_2022: -22.74% / -0.84 / 32.22% (MaxDD above 30% threshold)
- test_2023_2024: 19.64% / 0.46 / 38.46% (weak walk-forward performance)
- Proxy MaxDD: 51.9% (too high)

Root cause: imitation bonus (0.20) was too strong, causing overfitting to rulebase behavior.

## Hypothesis

Reducing imitation bonus + tightening drawdown controls will:
1. Reduce overfitting to training period
2. Bring ALL test periods' MaxDD below 30%
3. Improve walk-forward 2023-2024 excess returns

## Mechanism Delta (vs imitation_v1)

| Parameter | v1 | v2 |
|-----------|-----|-----|
| rule_imitation_bonus | 0.20 | 0.10 |
| rule_decision_bonus | 0.30 | 0.15 |
| episode_drawdown_penalty | 2.0 | 3.5 |
| episode_drawdown_threshold | 0.15 | 0.10 |
| downside_asymmetry_coef | 1.3 | 1.5 |
| drawdown_penalty | 0.15 | 0.30 |
| downside_penalty | 0.10 | 0.20 |
| risk_budget_penalty | 0.15 | 0.25 |
| micro_rebalance_penalty | 0.35 | 0.50 |
| transaction_cost | 0.08 | 0.10 |
| promotion_gate.max_test_2022_drawdown | 35.0 | 30.0 |

## Acceptance Criteria

- test_2022 MaxDD < 30%
- test_2023_2024 return > 30% (beat v1's 19.6%)
- test_2023_2024 Sharpe > 0.60
- Proxy Sharpe > 0.85
- Proxy MaxDD < 50%

## Failure Mode

- If drawdown controls are too tight, model may become permanently conservative and returns collapse
- If imitation bonus is too low, model may not learn rulebase timing at all
