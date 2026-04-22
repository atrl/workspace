# balanced_rule_residual_imitation_v1

## Motivation

Models consistently underperform the rulebase on raw return (v16: 632.4% vs rulebase: 688.6%).
The core problem is that alpha reward (`step_return - rule_return`) only gives positive signal
when the model **beats** the rulebase. Since the rulebase is already strong, the model gets
sparse positive rewards and frequent negatives, forcing it into overly conservative behavior.

## Hypothesis

Adding a **dense positive baseline** via rulebase imitation bonus will:
1. Provide stable positive rewards during training (even when model just matches rulebase)
2. Help the model learn rulebase timing first
3. Allow alpha residual to then push the model to find improvements

This is essentially a lightweight form of imitation learning + RL residual.

## External Reference

- AlphaGo training: first behavior cloning on human games, then RL self-play refinement
- RL with expert demonstrations: adding a small KL/matching term to the reward helps exploration

## Mechanism Delta (vs moderate_v1)

| Component | moderate_v1 | imitation_v1 |
|-----------|-------------|--------------|
| Branch | `use_moderate_reward: true` | `use_moderate_reward: true` |
| Position match bonus | 0 | `rule_imitation_bonus: 0.20` |
| Decision point bonus | 0 | `rule_decision_bonus: 0.30` |
| Environment state | `rule_pos` only | `rule_pos` + `prev_rule_pos` |

## Target Windows

- proxy (train): expect return closer to rulebase (~650-700%), Sharpe >= 0.85, MaxDD < 55%
- test_2022: MaxDD < 35%, avg_pos < 0.35

## Acceptance Criteria

- proxy return > 650% (beat v16 by > 18pp, close gap to rulebase)
- proxy Sharpe > 0.85
- proxy MaxDD < 55%
- test_2022 MaxDD < 35%

## Expected Trade-off

- May see slightly higher MaxDD than v16 (since we're encouraging more rulebase-like exposure)
- Should see higher return and better learning stability

## Failure Mode

- If imitation bonus is too strong, model may just copy rulebase and not explore improvements
- If too weak, no effect
- If rulebase has bad decisions in certain regimes, model may learn them too

## Comparison Row Key

`imitation_v1` — rulebase imitation bonus experiment
