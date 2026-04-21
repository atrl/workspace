# balanced_rule_residual_v7 — Notes

## Direction
User pivot: stop trade compression, focus on **drawdown control + Kelly expectancy** via holistic optimization (model + reward + features).

> "交易多不影响，主要要控制回测和凯利期望。优化的方式应该是模型、奖励、特征结合的结果。"

## Hypothesis
A larger model with stronger drawdown/downside penalties and tighter position caps will learn to better control tail risk while maintaining alpha. VIX features are already in the 59-dim observation — the model just needs more capacity and stronger reward shaping to use them for drawdown avoidance.

## Changes from v5 (baseline)
| Parameter | v5 | v7 | Rationale |
|-----------|-----|-----|-----------|
| hidden_sizes | [128, 64] | [256, 128] | More capacity to learn VIX→drawdown relationships |
| drawdown_penalty | 0.28 | 0.50 | Much stronger drawdown aversion |
| downside_penalty | 0.20 | 0.35 | Stronger penalty for downside vol |
| risk_budget_penalty | 0.24 | 0.35 | Tighter risk budget enforcement |
| high_exposure_penalty | 0.08 | 0.15 | Discourage high leverage in stress |
| return_scale | 1.15 | 1.20 | Slightly more focus on returns |
| position_cap.start_drawdown | 0.10 | 0.08 | Start capping earlier |
| position_cap.full_drawdown | 0.30 | 0.22 | Fully cap sooner |
| position_cap.min_cap | 0.50 | 0.40 | More aggressive cap under stress |
| flat_trade_penalty | 3.0 | 0.0 | Remove — trade count irrelevant |
| trade_lock_days | 12 | 8 | Reduce lock since trade count not priority |
| base_threshold | 0.06 | 0.06 | Restore normal (v6 had 0.10) |

## Expected Impact
- **MaxDD target**: < 50% (down from 55.9%)
- **test_2022 MaxDD target**: < 30% (down from 32.74%)
- **Sharpe target**: 0.90+ (maintain or improve)
- **Return target**: 700-800% (maintain)
- **Trade count**: Not a target — whatever the model learns

## Risk
- Over-penalizing drawdown may cause the agent to be too conservative, missing rallies
- Larger model may overfit if not enough data diversity
- Tighter caps may reduce returns in bull markets

## Baseline
- `balanced_rule_residual_v5`: return 780.7% / Sharpe 0.93 / MaxDD 55.9% / trades 326
- `balanced_rule_residual_v2`: valid score 104.61 @40, test 521.3%/Sharpe 0.86/MaxDD 50.4%/trades 316
