# TRANS_TE_ZSCORE

**方向**: factor
**类别**: mean-reversion
**状态**: 🟡 IC only

## 假设

Tracking error deviation unsustainable

## 方法/配置

公式: `tracking_error z-score 21d`

## 过程

通过 IC 检验和 standalone backtest 验证。

## 结果

| 指标 | 数值 |
|------|------|
| IC(h=5) | +0.0724 |

## 结论

IC=+0.0724 验证有效，但缺少 standalone backtest。

## 与基线/相关实验的差异

Transition 信号专门检测 stress release，与 core features 互补。

## 下一步

补充更多 horizon 的 backtest。
