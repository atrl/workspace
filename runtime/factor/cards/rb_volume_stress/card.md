# RB_VOLUME_STRESS

**方向**: factor
**类别**: liquidity
**状态**: 🟡 Weak

## 假设

预期方向: negative (horizon=5d)

市场逻辑: 成交量异动 × 波动压力得分，高值暗示情绪极端

## 方法/配置

公式: `volume_shock * vol_stress_score`

来源: rulebase_enhanced_panic
优先级: medium
风险: 高成交量可能伴随趋势延续

## 过程

通过 standalone signal backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|
| Best Sharpe | +0.198 |
| Profit Factor | 1.132 |
| Signal Accuracy | 0.552 |
| Num Signals | 120 |
| Turnover | 0.0795 |

## 结论

Standalone backtest Sharpe=+0.198，信号方向正确但不够强。需要与其他因子组合使用。

## 与基线/相关实验的差异

与基线或其他因子的差异待补充。

## 下一步

需进一步验证或与其他因子组合。
