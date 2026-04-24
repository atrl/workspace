# balanced_conservative_ft

**方向**: model
**类别**: conservative
**状态**: 🟡 Good

## 假设

在回报和风险控制之间寻找平衡，避免过度偏向某一市场体制。

## 方法/配置

Profile: `balanced_conservative_ft`
网络: GRU-64 + [128,64] MLP

## 过程

训练 → validation → checkpoint 选择 → test 回测 → promotion gate 评估。

## 结果

| 指标 | 数值 |
|------|------|
| Return | 660 |
| Sharpe | 0.897 |
| MaxDD | 50.787 |
| AvgPos | 0.553 |
| StressAvgPos | 0.303 |
| 2022 Return | -24.408 |
| 2022 MaxDD | — |
| 2020 Return | 120 |
| 2023-24 Return | 61.446 |
| Fixed Split Pass | ❌ |
| Walk Forward Pass | ❌ |

## 结论

表现尚可但不够强。Sharpe=0.897，MaxDD=50.787%。未通过 promotion gate。

## 与基线/相关实验的差异

与 canonical `balanced` 基线的差异：具体差异待补充。


## 实验原始记录

详细配置、动机和迭代历史见：
- `experiments/candidate/balanced_conservative_ft/notes.md`
- `experiments/candidate/balanced_conservative_ft/expectations.md`
- `experiments/candidate/balanced_conservative_ft/profile.yaml`
## 下一步

- 分析未通过 gate 的原因
- 考虑与其他机制组合
