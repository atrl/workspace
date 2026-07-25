# Failure modes

- 开发期与 outer 分离：开发期财富提升 2.9340%，但冻结 outer 相对原 Stage1 变化 -11.4214%。
- 年度相对门：候选仅 2/4 年跑赢 SOXX，要求至少 3/4。
- 动作漂移：增强分类器在 outer 的固定概率阈值产生明显年份动作稀疏差异；这说明分类概率尺度和标签关系都不能视为跨年稳定。
- 分位阈值诊断：只用历年训练分数校准可增加动作，但 2025 新增动作显著伤害收益，因此问题不只是概率压缩，不能靠调阈值修复。
- 结论：`rejected_feature_recovery_outer_generalization_failed`；outer 只做冻结否决，不反向选择候选。

## Outer recovery actions

| year | selected actions | baseline actions |
| ---: | ---: | ---: |
| 2023 | 0 | 3 |
| 2024 | 2 | 8 |
| 2025 | 0 | 14 |
| 2026 | 6 | 7 |

## Next objective

1. 预测扣除 10bp 成本后的连续 counterfactual advantage，不再只预测单日正负标签。
2. 独立预测动作后五日最差收益或尾部违约概率，作为 fail-closed veto。
3. 在开发期内部做嵌套 walk-forward 校准；保持 SOXL/SOXX/cash 合法投影、回撤控制器和 outer 边界不变。
