# balanced_floor40_sym_velocity_rules

**方向**: model
**类别**: execution_overlay
**状态**: 🟡 Good

## 假设

在回报和风险控制之间寻找平衡，避免过度偏向某一市场体制。

## 方法/配置

Profile: `balanced_floor40_sym_velocity_rules`
网络: GRU-64 + [128,64] MLP

## 过程

训练 → validation → checkpoint 选择 → test 回测 → promotion gate 评估。

## 结果

| 指标 | 数值 |
|------|------|
| Return | 386 |
| Sharpe | 0.800 |
| MaxDD | 62.336 |
| AvgPos | 0.456 |
| StressAvgPos | 0.317 |
| 2022 Return | -40.889 |
| 2022 MaxDD | — |
| 2020 Return | 112 |
| 2023-24 Return | 27.998 |
| Fixed Split Pass | ❌ |
| Walk Forward Pass | ❌ |

## 结论

表现尚可但不够强。Sharpe=0.800，MaxDD=62.336%。未通过 promotion gate。

## 与基线/相关实验的差异

与 canonical `balanced` 基线的差异：执行层 overlay 变化（position cap / frequency / mode routing）。

## 下一步

- 分析未通过 gate 的原因
- 考虑与其他机制组合
