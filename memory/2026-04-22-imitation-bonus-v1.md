# 2026-04-22: Rulebase Imitation Bonus v1

## Question
针对跑不过rulebase的问题，如果在rulebase的"埋点"上给超额奖励，能提高学习rulebase的能力吗？

## Analysis
- Rulebase: 688.6% / 0.824 / 69.88% (45 trades)
- v16 best model: 632.4% / 0.91 / 50.1%
- Gap: model underperforms by 56pp on raw return

Root cause: alpha reward (`step_return - rule_return`) only gives positive signal when model BEATS rulebase. Since rulebase is already strong, model gets sparse positive rewards → forced conservatism.

## Solution: Imitation Learning + RL Residual

Two new reward terms added to `_moderate_reward` branch:

1. **Position Match Bonus** (`rule_imitation_bonus: 0.20`)
   - `reward += (1.0 - |pos - rule_pos|) * 0.20`
   - Dense positive baseline: model gets reward just for being close to rulebase

2. **Decision Point Bonus** (`rule_decision_bonus: 0.30`)
   - When rulebase turns over > 3%, if model trades in same direction:
   - `reward += 0.30 * min(1.0, rule_turnover / 0.10)`
   - Extra reward at rulebase decision points

## Code Changes
- `src/soxl_quant/policy/environment.py`: added `prev_rule_pos` to state
- `src/soxl_quant/policy/reward.py`: added `_moderate_reward` method with imitation bonus
- `src/soxl_quant/config.py`: added `use_moderate_reward`, `rule_imitation_bonus`, `rule_decision_bonus` to RewardConfig
- `experiments/candidate/balanced_rule_residual_imitation_v1/profile.yaml`: new profile
- `configs/soxl/profile_registry.yaml`: registered new profile

## Training
- Started: 2026-04-22 14:57
- Command: `soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_imitation_v1 --skip-data-refresh --seed 20260422`
- Status: running (iter 8/20 stage 1 at time of writing)

## Acceptance Criteria
- proxy return > 650% (close gap to rulebase 688%)
- proxy Sharpe > 0.85
- proxy MaxDD < 55%
- test_2022 MaxDD < 35%
