# RB_RSI6_OVERSOLD

**方向**: factor
**类别**: momentum
**状态**: ❌ Invalid

## 假设

预期方向: negative (horizon=5d)

市场逻辑: RSI6超卖程度量化，0-1区间，越接近1超卖越严重

## 方法/配置

公式: `(30 - rsi_6 * 100).clip(lower=0) / 30.0`

来源: rulebase_panic_calculator
优先级: medium
风险: 暴跌中RSI可能长期钝化

## 过程

通过 standalone signal backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|
| Best Sharpe | -0.158 |
| Profit Factor | 1.423 |
| Signal Accuracy | 0.592 |
| Num Signals | 78 |
| Turnover | 0.0517 |

## 结论

Standalone backtest Sharpe=-0.158，信号无效。可能因市场结构变化或假设不成立。

## 与基线/相关实验的差异

与基线或其他因子的差异待补充。

## 下一步

需进一步验证或与其他因子组合。
