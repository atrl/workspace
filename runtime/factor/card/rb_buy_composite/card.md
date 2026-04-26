# RB_BUY_COMPOSITE

**方向**: factor
**类别**: composite
**状态**: 🟡 Weak

## 假设

预期方向: negative (horizon=5d)

市场逻辑: rulebase买入条件的综合信号：恐慌 + 日跌 + 累积跌

## 方法/配置

公式: `panic_score_proxy * 0.3 + daily_drop_severity * 0.4 + cum_drop_severity * 0.3`

来源: rulebase_buy_strategy
优先级: high
风险: 复合信号可能过度拟合历史买入点

## 过程

通过 standalone signal backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|
| Best Sharpe | +0.115 |
| Profit Factor | 1.110 |
| Signal Accuracy | 0.552 |
| Num Signals | 290 |
| Turnover | 0.1922 |

## 结论

Standalone backtest Sharpe=+0.115，信号方向正确但不够强。需要与其他因子组合使用。


## 参考文献

- rulebase 启发: 复合买入信号
## 与基线/相关实验的差异

与基线或其他因子的差异待补充。

## 下一步

需进一步验证或与其他因子组合。
