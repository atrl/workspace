# TRANS_VIX_DIVERGENCE

**方向**: factor
**类别**: mean-reversion
**状态**: ✅ Strong

## 假设

VIX falls but price near lows → sentiment fading

## 方法/配置

公式: `vix_fall_pct * price_near_low`

## 过程

通过 IC 检验和 standalone backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|
| Best Sharpe | +1.041 |
| Profit Factor | 1.427 |
| Signal Accuracy | 0.642 |
| Num Signals | 58 |

## 结论

Sharpe=+1.041，transition 信号非常强。

## 与基线/相关实验的差异

Transition 信号专门检测 stress release，与 core features 互补。

## 下一步

-
