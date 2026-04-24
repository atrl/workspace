# balanced_kelly_execution_v8

**方向**: model
**类别**: kelly
**状态**: 🟡 Good

## 假设

在回报和风险控制之间寻找平衡，避免过度偏向某一市场体制。

## 方法/配置

Profile: `balanced_kelly_execution_v8`
网络: GRU-64 + [128,64] MLP

## 过程

训练 → validation → checkpoint 选择 → test 回测 → promotion gate 评估。

## 结果

| 指标 | 数值 |
|------|------|
| Return | 682 |
| Sharpe | 0.898 |
| MaxDD | 60.344 |
| AvgPos | 0.564 |
| StressAvgPos | 0.349 |
| 2022 Return | -24.576 |
| 2022 MaxDD | — |
| 2020 Return | 107 |
| 2023-24 Return | 45.613 |
| Fixed Split Pass | ❌ |
| Walk Forward Pass | ❌ |

## 结论

表现尚可但不够强。Sharpe=0.898，MaxDD=60.344%。未通过 promotion gate。

## 与基线/相关实验的差异

与 canonical `balanced` 基线的差异：奖励函数包含 Kelly 项，与纯 log 奖励不同。

## 下一步

- 分析未通过 gate 的原因
- 考虑与其他机制组合
