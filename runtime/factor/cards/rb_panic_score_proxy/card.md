# RB_PANIC_SCORE_PROXY

**方向**: factor
**类别**: sentiment
**状态**: 🟡 Weak

## 假设

预期方向: negative (horizon=5d)

市场逻辑: VIX驱动的恐慌情绪(sigmoid映射) × 成交量放大器 + RSI6超卖调节

## 方法/配置

公式: `sigmoid_vix_zscore * relative_volume_zscore + rsi6_adjustment`

来源: rulebase_panic_calculator
优先级: high
风险: VIX数据缺失时退化为vol_stress_score

## 过程

通过 standalone signal backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|
| Best Sharpe | +0.172 |
| Profit Factor | 1.130 |
| Signal Accuracy | 0.549 |
| Num Signals | 123 |
| Turnover | 0.0815 |

## 结论

Standalone backtest Sharpe=+0.172，信号方向正确但不够强。需要与其他因子组合使用。


## 参考文献

- rulebase 启发: EnhancedPanicCalculator
## 与基线/相关实验的差异

与基线或其他因子的差异待补充。

## 下一步

需进一步验证或与其他因子组合。
