# balanced_rule_residual_imitation_v4

**方向**: model
**类别**: imitation
**状态**: 🟡 Good

## 假设

在回报和风险控制之间寻找平衡，避免过度偏向某一市场体制。

## 方法/配置

Profile: `balanced_rule_residual_imitation_v4`
网络: GRU-64 + [128,64] MLP

## 过程

训练 → validation → checkpoint 选择 → test 回测 → promotion gate 评估。

## 结果

| 指标 | 数值 |
|------|------|
| Return | 586 |
| Sharpe | 0.924 |
| MaxDD | 50.949 |
| AvgPos | 0.470 |
| StressAvgPos | 0.294 |
| 2022 Return | -21.881 |
| 2022 MaxDD | — |
| 2020 Return | 110 |
| 2023-24 Return | 43.720 |
| Fixed Split Pass | ❌ |
| Walk Forward Pass | ❌ |

## 结论

表现尚可但不够强。Sharpe=0.924，MaxDD=50.949%。未通过 promotion gate。

## 与基线/相关实验的差异

与 canonical `balanced` 基线的差异：包含 rulebase 模仿奖励，收敛更快但可能过拟合历史买入点。

## 下一步

- 分析未通过 gate 的原因
- 考虑与其他机制组合
