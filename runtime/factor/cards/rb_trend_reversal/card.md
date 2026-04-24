# RB_TREND_REVERSAL

**方向**: factor
**类别**: trend
**状态**: ⚪ Unverified

## 假设

预期方向: negative (horizon=10d)

市场逻辑: 长期均线偏离 × 波动率放大，极端负值暗示反转

## 方法/配置

公式: `anchor_gap_180 * (1 + realized_vol_21)`

来源: rulebase_buy_strategy
优先级: medium
风险: 趋势延续时间不确定

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
