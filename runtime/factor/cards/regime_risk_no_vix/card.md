# regime_risk_no_vix

**方向**: factor
**类别**: composite
**状态**: 🟡 Ablated

## 假设

VIX-free stress index

## 方法/配置

公式: `vol_stress + vol + dd + downside + semivol + te_vol + loss_streak`

## 过程

VIX 缺失时的降级方案。

## 结果

| 指标 | 数值 |
|------|------|
| Max Correlation | 0.902 (with realized_vol_21) |

## 结论

核心 composite 特征，作为 ablation 备用。

## 与基线/相关实验的差异

与单一因子的差异在于综合多个信号。

## 下一步

持续监控与其他特征的相关性。
