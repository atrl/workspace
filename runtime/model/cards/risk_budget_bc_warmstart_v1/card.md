# risk_budget_bc_warmstart_v1

**方向**: model
**类别**: risk_budget
**状态**: 🟡 Good

## 假设

通过风险预算分配机制，限制单一标的风险暴露。

## 方法/配置

Profile: `risk_budget_bc_warmstart_v1`
网络: GRU-64 + [128,64] MLP

## 过程

训练 → validation → checkpoint 选择 → test 回测 → promotion gate 评估。

## 结果

| 指标 | 数值 |
|------|------|
| Return | 251 |
| Sharpe | 0.911 |
| MaxDD | 27.266 |
| AvgPos | 0.266 |
| StressAvgPos | 0.100 |
| 2022 Return | -5.038 |
| 2022 MaxDD | — |
| 2020 Return | 36.039 |
| 2023-24 Return | 48.955 |
| Fixed Split Pass | ❌ |
| Walk Forward Pass | ❌ |

## 结论

表现尚可但不够强。Sharpe=0.911，MaxDD=27.266%。未通过 promotion gate。

## 与基线/相关实验的差异

与 canonical `balanced` 基线的差异：引入风险预算约束，avg_pos 通常更低。


## 实验原始记录

详细配置、动机和迭代历史见：
- `experiments/candidate/risk_budget_usage_v1/notes.md`
- `experiments/candidate/risk_budget_usage_v1/expectations.md`
- `experiments/candidate/risk_budget_usage_v1/profile.yaml`
## 下一步

- 分析未通过 gate 的原因
- 考虑与其他机制组合
