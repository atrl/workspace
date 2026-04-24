# smoke_tide

**方向**: model
**类别**: smoke
**状态**: 🟡 Moderate

## 假设

快速验证基础训练流程和特征工程是否正常工作。

## 方法/配置

Profile: `smoke_tide`
网络: GRU-64 + [128,64] MLP

## 过程

训练 → validation → checkpoint 选择 → test 回测 → promotion gate 评估。

## 结果

| 指标 | 数值 |
|------|------|
| Return | 307 |
| Sharpe | 0.693 |
| MaxDD | 70.600 |
| AvgPos | 0.615 |
| StressAvgPos | 0.380 |
| 2022 Return | -48.912 |
| 2022 MaxDD | — |
| 2020 Return | 140 |
| 2023-24 Return | 31.097 |
| Fixed Split Pass | ❌ |
| Walk Forward Pass | ❌ |

## 结论

表现尚可但不够强。Sharpe=0.693，MaxDD=70.600%。未通过 promotion gate。

## 与基线/相关实验的差异

与 canonical `balanced` 基线的差异：smoke 是快速验证配置，通常使用更少特征和更短训练。

## 下一步

- 分析未通过 gate 的原因
- 考虑与其他机制组合
