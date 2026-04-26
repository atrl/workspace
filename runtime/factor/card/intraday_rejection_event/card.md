# INTRADAY_REJECTION_EVENT

**方向**: factor
**类别**: transition / intraday-reversal
**状态**: 🟡 Candidate / blocked

## 假设

币圈单资产中常见的 liquidation wick / panic rejection 可以映射到 SOXL：
大振幅日出现长下影、收盘靠近高位且放量，代表盘中卖压被吸收，后续可能反弹。

## 方法/配置

公式: `lower_wick_ratio * close_location * positive_intraday_range_zscore * (1 + positive_relative_volume_zscore)`

## 结果

| 指标 | 数值 |
|------|------|
| Best IC Horizon | 20d |
| IC Mean | +0.015727 |
| IC Positive Ratio | 0.215 |
| IC t-stat | 2.428 |
| Threshold | 0.00 |
| Sharpe | +0.261 |
| Profit Factor | 1.078 |
| Signal Accuracy | 0.529 |
| Num Signals | 1367 |
| Max Core Correlation | 0.444 |

## 结论

该因子先作为 standalone candidate 留存。只有在 IC、方向准确率、profit factor
和相关性预算同时通过后，才允许进入 RL feature promotion 讨论。

## 参考

- Crypto single-asset intraday reversal / liquidation wick analogue
- 与 `ACAD_INTRADAY_REVERSAL_21` 区分：本因子是事件型 wick rejection，不是 21d 平滑均值。
