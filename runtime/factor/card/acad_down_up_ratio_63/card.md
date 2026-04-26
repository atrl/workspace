# ACAD_DOWN_UP_RATIO_63

**方向**: factor
**类别**: mean-reversion
**状态**: 🟡 Weak

## 假设

Down/up ratio 63d，预期方向: negative

## 方法/配置

公式: `down/up vol ratio 63d`

## 过程

通过 standalone signal backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|
| Best Sharpe | +0.129 |
| Profit Factor | 1.159 |
| Signal Accuracy | 0.550 |
| Num Signals | 8 |

## 结论

Sharpe=+0.129，有一定效果但不够强。


## 参考文献

- Qiao et al. (2022): upside/downside realized semi-variances 启发
## 与基线/相关实验的差异

学术因子与 rulebase/transition 因子的差异待补充。

## 下一步

需进一步验证。
