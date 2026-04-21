# balanced_rule_residual_v6 — Notes

## Hypothesis
Flat penalties (0.5→3.0) failed to compress trades because they are soft costs absorbed by the value function. The agent still trades ~326× because execution thresholds (base: 0.06) allow frequent small rebalances, and the inactivity penalty (0.80) actively punishes holding positions.

**New approach: structural constraint + reduced trading incentives**
1. Increase `base_threshold` 0.06→0.10 and `quiet_threshold` 0.08→0.12 — only rebalance on meaningful position changes (hard constraint)
2. Remove `inactivity_penalty` 0.80→0.0 — stop punishing the agent for not trading
3. Reduce `alpha_scale` 2.0→1.5 — lower reward for alpha-seeking trades
4. Keep `flat_trade_penalty: 3.0` as additional deterrent

## Changes from v5
| Parameter | v5 | v6 | Rationale |
|-----------|-----|-----|-----------|
| base_threshold | 0.06 | 0.10 | Hard filter: only rebalance on ≥10% position change |
| quiet_threshold | 0.08 | 0.12 | Same logic in quiet regime |
| inactivity_penalty | 0.80 | 0.0 | Stop punishing holding positions |
| alpha_scale | 2.0 | 1.5 | Reduce incentive for frequent alpha trades |
| flat_trade_penalty | 3.0 | 3.0 | Keep as additional cost |
| trade_lock_days | 12 | 12 | Keep |

## Expected Impact
- Trade count target: 120-180 (down from 326)
- Return target: 550-650% (may drop due to higher thresholds, but rule-based benchmark achieved 688.6% with 45 trades)
- Sharpe target: 0.85-0.90
- MaxDD target: 50-58%

## Risk
- Higher thresholds may cause the agent to miss sharp market turns
- Could become too sluggish in fast-moving markets
- If trades drop below 100, strategy may be too conservative

## Baseline
- `balanced_rule_residual_v2`: valid score 104.61 @40, test 521.3%/Sharpe 0.86/MaxDD 50.4%/trades 316
