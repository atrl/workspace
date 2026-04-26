# RB_DAILY_DROP_SEVERITY

**方向**: factor
**类别**: drawdown
**状态**: ⚪ Unverified

## 假设

预期方向: negative (horizon=3d)

市场逻辑: 日跌幅经波动率标准化，极端值暗示反转

## 方法/配置

公式: `daily_return / (vol_coef_21 + 1e-8)`

来源: rulebase_buy_strategy
优先级: high
风险: 高波动期可能连续下跌

## 过程

通过 standalone signal backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|

## 结论

尚未进行 standalone backtest 或 IC 验证。


## 参考文献

- rulebase 启发: buy strategy
## 与基线/相关实验的差异

与基线或其他因子的差异待补充。

## 下一步

需进一步验证或与其他因子组合。
