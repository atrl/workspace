# RB_VIX_TERM_STRESS

**方向**: factor
**类别**: sentiment
**状态**: ⚪ Unverified

## 假设

预期方向: negative (horizon=5d)

市场逻辑: VIX期限结构斜率 × 短期VIX比率，异常结构暗示压力

## 方法/配置

公式: `vix_term_slope * vix_short_ratio`

来源: rulebase_panic_calculator
优先级: medium
风险: 需要VIX3M/VIX9D数据

## 过程

通过 standalone signal backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|

## 结论

尚未进行 standalone backtest 或 IC 验证。


## 参考文献

- rulebase 启发: VIX 期限结构 stress
## 与基线/相关实验的差异

与基线或其他因子的差异待补充。

## 下一步

需进一步验证或与其他因子组合。
