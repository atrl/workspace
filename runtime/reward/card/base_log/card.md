# Base Log Reward

**方向**: reward
**类别**: base_reward
**状态**: ✅ Baseline

## 假设

纯对数回报奖励 `R_t = log(W_t/W_{t-1})` 作为基线，验证其单独驱动策略的能力。

## 奖励公式

```
R = log(portfolio_value_t / portfolio_value_{t-1})
```

## 参数与结果对比

| Profile | 额外机制 | Return | Sharpe | MaxDD | AvgPos | 状态 |
|---------|---------|--------|--------|-------|--------|------|
| balanced (canonical) | 无 | 97 | 2.16 | 21.1% | 0.52 | ✅ Strong |
| balanced_rule_residual_v19 | residual + penalty | 105 | 2.24 | 22.3% | 0.52 | ✅ Strong |
| smoke | 无 | 307 | 0.69 | 70.6% | 0.62 | 🟡 Moderate |
| risk_budget_usage_v1 | risk budget | 274 | 0.94 | 32.5% | 0.27 | 🟡 Good |

## 结论

纯 log 奖励本身不足以单独驱动复杂交易决策（smoke 表现弱）。但当配合惩罚项、residual 机制或 risk budget 时，可以作为有效基线。canonical balanced 的 Sharpe=2.16 是最佳基线。

## 与 Kelly/Imitation 的差异

- Kelly 在 log 基础上叠加 signal quality 项
- Imitation 在 log 基础上叠加 rulebase 埋点奖励
- Base log 更简洁，但可能收敛慢或探索不足

## 下一步

- [ ] 作为所有 reward ablation 的对照组
