# RB_RSI6_OVERBOUGHT

**方向**: factor
**类别**: momentum
**状态**: ⚪ Unverified

## 假设

预期方向: positive (horizon=5d)

市场逻辑: RSI6超买程度量化，0-1区间，越接近1超买越严重

## 方法/配置

公式: `(rsi_6 * 100 - 70).clip(lower=0) / 30.0`

来源: rulebase_panic_calculator
优先级: medium
风险: 强趋势中RSI可能长期高位

## 过程

通过 standalone signal backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|

## 结论

尚未进行 standalone backtest 或 IC 验证。


## 参考文献

- rulebase 启发: RSI6 超买
## 与基线/相关实验的差异

与基线或其他因子的差异待补充。

## 下一步

需进一步验证或与其他因子组合。
