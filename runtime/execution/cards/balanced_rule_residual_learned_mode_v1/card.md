# balanced_rule_residual_learned_mode_v1

**方向**: execution
**类别**: mode_routing
**状态**: 🟡 Good

## 假设

通过 regime mode routing 在不同市场体制下使用不同策略。

## 方法/配置

Profile: `balanced_rule_residual_learned_mode_v1`

## 过程

训练 → test 回测 → 与无 overlay 基线对比。

## 结果

| 指标 | 数值 |
|------|------|
| Return | 507 |
| Sharpe | 0.873 |
| MaxDD | 50.006 |
| AvgPos | 0.469 |
| 2022 MaxDD | — |

## 结论

Sharpe=0.873，MaxDD=50.006%。

## 与基线/相关实验的差异

与无 overlay 基线（如 balanced）的差异：balanced 基线 Sharpe=0.816，MaxDD=52.750%。

## 下一步

- 继续优化 overlay 参数
- 考虑动态频率调整
