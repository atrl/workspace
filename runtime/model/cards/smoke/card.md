# smoke

**方向**: model
**类别**: smoke
**状态**: 🟡 Moderate

## 假设

快速验证基础训练流程和特征工程是否正常工作。

## 方法/配置

Profile: `smoke`
网络: GRU-64 + [128,64] MLP

## 过程

训练 → validation → checkpoint 选择 → test 回测 → promotion gate 评估。

## 结果

| 指标 | 数值 |
|------|------|
| Return | 298 |
| Sharpe | 0.683 |
| MaxDD | 76.832 |
| AvgPos | 0.727 |
| StressAvgPos | 0.415 |
| 2022 Return | -57.215 |
| 2022 MaxDD | — |
| 2020 Return | 175 |
| 2023-24 Return | 49.242 |
| Fixed Split Pass | ❌ |
| Walk Forward Pass | ❌ |

## 结论

表现尚可但不够强。Sharpe=0.683，MaxDD=76.832%。未通过 promotion gate。

## 与基线/相关实验的差异

与 canonical `balanced` 基线的差异：smoke 是快速验证配置，通常使用更少特征和更短训练。

## 下一步

- 分析未通过 gate 的原因
- 考虑与其他机制组合
