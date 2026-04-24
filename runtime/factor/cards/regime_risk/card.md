# regime_risk

**方向**: factor
**类别**: composite
**状态**: ✅ Core

## 假设

Multi-signal stress index with VIX

## 方法/配置

公式: `vix% + vol + dd + downside + semivol + te_vol + loss_streak`

## 过程

训练核心特征，用于 regime 分类和 risk 调整。

## 结果

| 指标 | 数值 |
|------|------|
| Max Correlation | 0.896 (with downside_semivol_21) |

## 结论

核心 composite 特征，用于训练。

## 与基线/相关实验的差异

与单一因子的差异在于综合多个信号。

## 下一步

持续监控与其他特征的相关性。
