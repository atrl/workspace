# balanced

**方向**: model
**类别**: balanced
**状态**: ❌ Weak

## 假设

在回报和风险控制之间寻找平衡，避免过度偏向某一市场体制。

## 方法/配置

Profile: `balanced`
网络: GRU-64 + [128,64] MLP

## 过程

训练 → validation → checkpoint 选择 → test 回测 → promotion gate 评估。

## 结果

| 指标 | 数值 |
|------|------|
| Return | 1.000 |
| Sharpe | 0.500 |
| MaxDD | — |
| AvgPos | — |
| StressAvgPos | — |
| 2022 Return | 1.000 |
| 2022 MaxDD | — |
| 2020 Return | — |
| 2023-24 Return | — |
| Fixed Split Pass | ❌ |
| Walk Forward Pass | ❌ |

## 结论

表现较弱。Sharpe=0.500，需要重新设计。

## 与基线/相关实验的差异

与 canonical `balanced` 基线的差异：同属 balanced 系列，具体参数差异见方法配置。


## 实验原始记录

详细配置、动机和迭代历史见：
- `experiments/canonical/balanced/notes.md`
- `experiments/canonical/balanced/expectations.md`
- `experiments/canonical/balanced/profile.yaml`
## 下一步

- 重新评估假设
- 考虑放弃或大幅修改
