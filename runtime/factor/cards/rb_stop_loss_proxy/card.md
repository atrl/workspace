# RB_STOP_LOSS_PROXY

**方向**: factor
**类别**: drawdown
**状态**: ⚪ Unverified

## 假设

预期方向: positive (horizon=1d)

市场逻辑: 接近-20%止损线的程度，高值暗示止损风险

## 方法/配置

公式: `(cum_drop_30 + 0.20).clip(lower=0, upper=0.20) / 0.20`

来源: rulebase_sell_strategy
优先级: medium
风险: 止损后可能立即反弹

## 过程

通过 standalone signal backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|

## 结论

尚未进行 standalone backtest 或 IC 验证。


## 参考文献

- rulebase 启发: 止损代理
## 与基线/相关实验的差异

与基线或其他因子的差异待补充。

## 下一步

需进一步验证或与其他因子组合。
