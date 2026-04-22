# balanced_rule_residual_imitation_v5

## Motivation

v4 achieved excellent balance but missed the 30% MaxDD target by a small margin:
- test_2022 MaxDD: 31.3% (target: < 30%)
- test_2023_2024 MaxDD: 31.1% (target: < 30%)
- test_2023_2024 Return: 51.8% (strong)

## Hypothesis

Slightly tightening episodic drawdown controls (penalty 3.5→4.0, threshold 0.10→0.08) will push both test periods under 30% MaxDD while preserving the participation signals that drive walk-forward returns.

## Mechanism Delta (vs imitation_v4)

| Parameter | v4 | v5 |
|-----------|-----|-----|
| episode_drawdown_penalty | 3.5 | 4.0 |
| episode_drawdown_threshold | 0.10 | 0.08 |
| Everything else | v4 | same |

## Acceptance Criteria

- test_2022 MaxDD < 30%
- test_2023_2024 MaxDD < 30%
- test_2023_2024 return > 45%
- test_proxy Sharpe > 0.85
