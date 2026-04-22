# balanced_rule_residual_imitation_v7

## Motivation

After 6 versions, the pattern is clear:
- v2: Best MaxDD control (30.2% / 29.4%) but weak returns (34.0%)
- v4: Best returns (51.8%) but MaxDD slightly over 30% (31.3% / 31.1%)
- v5/v6: Attempts to tighten controls further were counterproductive

## Hypothesis

A compromise between v2's proven drawdown framework and v4's participation signals will achieve both goals:
- Keep v2's drawdown parameters (episode_penalty=3.5, step_penalty=0.30)
- Use moderate participation signals (trend_alignment=0.32, opportunity_cost=0.15)
- This should keep MaxDD < 30% while improving walk-forward returns toward v4 levels

## Mechanism Delta (vs v2/v4)

| Parameter | v2 | v4 | v7 |
|-----------|-----|-----|-----|
| episode_drawdown_penalty | 3.5 | 3.5 | 3.5 |
| drawdown_penalty | 0.30 | 0.30 | 0.30 |
| trend_alignment_bonus | 0.25 | 0.40 | 0.32 |
| opportunity_cost_penalty | 0.05 | 0.20 | 0.15 |
| alpha_scale | 3.0 | 3.2 | 3.2 |

## Acceptance Criteria

- test_2022 MaxDD < 30%
- test_2023_2024 MaxDD < 30%
- test_2023_2024 return > 40%
