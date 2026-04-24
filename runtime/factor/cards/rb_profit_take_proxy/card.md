# RB_PROFIT_TAKE_PROXY

**方向**: factor
**类别**: drawdown
**状态**: ⚪ Unverified

## 假设

预期方向: positive (horizon=3d)

市场逻辑: 从深度回撤中恢复的程度，接近1暗示获利了结

## 方法/配置

公式: `(cum_drop_30 + 0.30).clip(lower=0, upper=0.60) / 0.60`

来源: rulebase_sell_strategy
优先级: medium
风险: 仅反映价格恢复，非真实持仓盈亏

## 过程

通过 standalone signal backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|

## 结论

尚未进行 standalone backtest 或 IC 验证。

## 与基线/相关实验的差异

与基线或其他因子的差异待补充。

## 下一步

需进一步验证或与其他因子组合。
