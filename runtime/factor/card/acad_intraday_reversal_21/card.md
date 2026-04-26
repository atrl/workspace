# ACAD_INTRADAY_REVERSAL_21

**方向**: factor
**类别**: mean-reversion
**状态**: 🟡 Weak

## 假设

Intraday reversal 21d，预期方向: negative

## 方法/配置

公式: `(close-open)/(high-low) rolling 21`

## 过程

通过 standalone signal backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|
| Best Sharpe | +0.326 |
| Profit Factor | 1.210 |
| Signal Accuracy | 0.535 |
| Num Signals | 28 |
| Max Correlation | 0.729 (with sharpe_proxy_21) |

## 结论

Sharpe=+0.326，有一定效果但不够强。


## 参考文献

- 原创: intraday momentum reversal (orthogonal to existing features)
## 与基线/相关实验的差异

学术因子与 rulebase/transition 因子的差异待补充。

## 下一步

需进一步验证。
