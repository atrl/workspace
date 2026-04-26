# Imitation Reward

**方向**: reward
**类别**: imitation_reward
**状态**: 🟡 Incremental

## 假设

通过 rulebase 策略的埋点奖励（imitation bonus），引导 RL agent 学习人类可解释的交易行为，加速收敛。

## 奖励公式

```
R = R_log + imitation_bonus * rulebase_signal_strength
# imitation_bonus: 当 rulebase 买入/卖出条件触发时给予正奖励
```

## 参数与结果对比

| Profile | 变体 | Return | Sharpe | MaxDD | AvgPos | 状态 |
|---------|------|--------|--------|-------|--------|------|
| balanced_rule_residual_imitation_v6 | v6 | 768 | 0.93 | 55.5% | 0.55 | 🟡 Good |
| balanced_rule_residual_imitation_v2_fix | v2 fix | 741 | 0.92 | 59.1% | 0.55 | 🟡 Good |
| balanced_rule_residual_imitation_v3 | v3 | 796 | 0.92 | 59.8% | 0.59 | 🟡 Good |
| balanced_rule_residual_imitation_v4 | v4 | 586 | 0.92 | 50.9% | 0.47 | 🟡 Good |
| balanced_rule_residual_imitation_v5 | v5 | 668 | 0.92 | 56.6% | 0.53 | 🟡 Good |
| balanced_rule_residual_imitation_v2 | v2 | 525 | 0.91 | 50.5% | 0.46 | 🟡 Good |
| balanced_rule_residual_imitation_v7 | v7 | 634 | 0.89 | 55.8% | 0.53 | 🟡 Good |
| balanced_rule_residual_imitation_v1 | v1 | 561 | 0.87 | 54.3% | 0.54 | 🟡 Good |

## 结论

Imitation reward 有效，v1-v7 的 Sharpe 在 0.87-0.93 之间，表现稳定。v1 曾达到 812% proxy return（历史最佳）。但随着版本迭代，没有持续提升——v6 的 Sharpe=0.93 与 v2 的 0.91 差距不大，说明 imitation bonus 的边际收益在递减。

## 与 Base Log 的差异

- 强制 agent 模仿 rulebase 行为，减少了探索空间
- 可能过拟合历史买入点（rulebase 信号基于历史阈值）
- 收敛更快，但泛化能力可能受限

## 下一步

- [ ] 分析 imitation 是否导致 2022 stress 态表现弱化
- [ ] 尝试动态 imitation weight（随训练进度衰减）
