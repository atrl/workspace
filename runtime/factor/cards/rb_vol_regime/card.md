# RB_VOL_REGIME

**方向**: factor
**类别**: volatility
**状态**: ⚪ Unverified

## 假设

预期方向: negative (horizon=5d)

市场逻辑: 波动率系数 = HV_20 / 历史中位数，>1高波动 <1低波动

## 方法/配置

公式: `vol_coef_21`

来源: rulebase_execution
优先级: medium
风险: 高波动可能持续

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
