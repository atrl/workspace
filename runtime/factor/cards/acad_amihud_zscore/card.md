# ACAD_AMIHUD_ZSCORE

**方向**: factor
**类别**: mean-reversion
**状态**: 🟡 Weak

## 假设

Amihud illiquidity z-score，预期方向: negative

## 方法/配置

公式: `price / (volume * spread)`

## 过程

通过 standalone signal backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|
| Best Sharpe | +0.284 |
| Profit Factor | 1.212 |
| Signal Accuracy | 0.568 |
| Num Signals | 6 |

## 结论

Sharpe=+0.284，有一定效果但不够强。

## 与基线/相关实验的差异

学术因子与 rulebase/transition 因子的差异待补充。

## 下一步

需进一步验证。
