# balanced_floor40_sym_tight_crash

**方向**: execution
**类别**: position_cap
**状态**: 🟡 Moderate

## 假设

通过 drawdown-based 动态仓位下限/上限控制 drawdown。

## 方法/配置

Profile: `balanced_floor40_sym_tight_crash`

## 过程

训练 → test 回测 → 与无 overlay 基线对比。

## 结果

| 指标 | 数值 |
|------|------|
| Return | 397 |
| Sharpe | 0.797 |
| MaxDD | 56.532 |
| AvgPos | 0.478 |
| 2022 MaxDD | — |

## 结论

Sharpe=0.797，MaxDD=56.532%。

## 与基线/相关实验的差异

与无 overlay 基线（如 balanced）的差异：balanced 基线 Sharpe=0.816，MaxDD=52.750%。

## 下一步

- 继续优化 overlay 参数
- 考虑动态频率调整
