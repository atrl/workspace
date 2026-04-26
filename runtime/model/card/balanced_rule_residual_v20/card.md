# balanced_rule_residual_v20

**方向**: model
**类别**: balanced
**状态**: 🟡 Good

## 假设

在回报和风险控制之间寻找平衡，避免过度偏向某一市场体制。

## 方法/配置

Profile: `balanced_rule_residual_v20`
网络: GRU-64 + [128,64] MLP

## 过程

训练 → validation → checkpoint 选择 → test 回测 → promotion gate 评估。

## 结果

| 指标 | 数值 |
|------|------|
| Return | 623 |
| Sharpe | 0.897 |
| MaxDD | 53.847 |
| AvgPos | 0.518 |
| StressAvgPos | 0.309 |
| 2022 Return | -23.432 |
| 2022 MaxDD | — |
| 2020 Return | 110 |
| 2023-24 Return | 42.573 |
| Fixed Split Pass | ❌ |
| Walk Forward Pass | ❌ |

## 结论

表现尚可但不够强。Sharpe=0.897，MaxDD=53.847%。未通过 promotion gate。

## 与基线/相关实验的差异

与 canonical `balanced` 基线的差异：同属 balanced 系列，具体参数差异见方法配置。

## 下一步

- 分析未通过 gate 的原因
- 考虑与其他机制组合
