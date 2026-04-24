# balanced_rule_residual_v19

**方向**: model
**类别**: balanced
**状态**: ❌ Weak

## 假设

在回报和风险控制之间寻找平衡，避免过度偏向某一市场体制。

## 方法/配置

Profile: `balanced_rule_residual_v19`
网络: GRU-64 + [128,64] MLP

## 过程

训练 → validation → checkpoint 选择 → test 回测 → promotion gate 评估。

## 结果

| 指标 | 数值 |
|------|------|
| Return | -27.269 |
| Sharpe | -0.920 |
| MaxDD | 36.322 |
| AvgPos | 0.298 |
| StressAvgPos | 0.293 |
| 2022 Return | -23.311 |
| 2022 MaxDD | — |
| 2020 Return | — |
| 2023-24 Return | — |
| Fixed Split Pass | ❌ |
| Walk Forward Pass | ❌ |

## 结论

表现较弱。Sharpe=-0.920，需要重新设计。

## 与基线/相关实验的差异

与 canonical `balanced` 基线的差异：同属 balanced 系列，具体参数差异见方法配置。

## 下一步

- 重新评估假设
- 考虑放弃或大幅修改
