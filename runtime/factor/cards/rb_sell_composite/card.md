# RB_SELL_COMPOSITE

**方向**: factor
**类别**: composite
**状态**: ⚪ Unverified

## 假设

预期方向: positive (horizon=3d)

市场逻辑: rulebase卖出条件的综合信号：超买 + 止盈 + 止损

## 方法/配置

公式: `rsi6_overbought * 0.4 + profit_position_proxy * 0.3 + stop_loss_proxy * 0.3`

来源: rulebase_sell_strategy
优先级: high
风险: 需要持仓状态信息，非纯价格特征

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
