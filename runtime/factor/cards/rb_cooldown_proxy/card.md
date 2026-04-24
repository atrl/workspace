# RB_COOLDOWN_PROXY

**方向**: factor
**类别**: timing
**状态**: ✅ Strong

## 假设

预期方向: negative (horizon=10d)

市场逻辑: 距离高点天数 × 脱离底部程度，高值暗示冷却期结束

## 方法/配置

公式: `days_since_high_63 * (1 - below_anchor_ratio_63)`

来源: rulebase_buy_strategy
优先级: low
风险: 间接信号，噪声较大

## 过程

通过 standalone signal backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|
| Best Sharpe | +0.938 |
| Profit Factor | 1.400 |
| Signal Accuracy | 0.588 |
| Num Signals | 32 |
| Turnover | 0.0212 |

## 结论

Standalone backtest Sharpe=+0.938，信号有效。Profit factor=1.40，准确率=0.588。

## 与基线/相关实验的差异

与基线或其他因子的差异待补充。

## 下一步

-
