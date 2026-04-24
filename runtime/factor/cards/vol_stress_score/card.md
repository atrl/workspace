# vol_stress_score

**方向**: factor
**类别**: composite
**状态**: 🟡 Ablated

## 假设

Price-only stress proxy

## 方法/配置

公式: `rv_z + dd_z + ir_z + vol_shock + kurt`

## 过程

纯价格数据的 stress 度量。

## 结果

| 指标 | 数值 |
|------|------|
| Max Correlation | 0.543 (with realized_vol_10) |

## 结论

核心 composite 特征，作为 ablation 备用。


## 参考文献

- arXiv 2510.03236 启发: price-only stress composite
## 与基线/相关实验的差异

与单一因子的差异在于综合多个信号。

## 下一步

持续监控与其他特征的相关性。
