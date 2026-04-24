# ACAD_VOLATILITY_DRAG

**方向**: factor
**类别**: momentum
**状态**: ❌ Invalid

## 假设

Volatility drag，预期方向: negative

## 方法/配置

公式: `decay_63 vs vol drag estimate`

## 过程

通过 standalone signal backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|
| Best Sharpe | -0.032 |
| Profit Factor | 1.086 |
| Signal Accuracy | 0.543 |
| Num Signals | 41 |

## 结论

Sharpe=-0.032，学术因子在 SOXL 上无效。

## 与基线/相关实验的差异

学术因子与 rulebase/transition 因子的差异待补充。

## 下一步

需进一步验证。
