# SIGNED_TE_REVERSION_10_STRESS_GATE

**方向**: factor
**类别**: transition / gated mean-reversion
**状态**: 🟡 Candidate / blocked

## 假设

`SIGNED_TE_REVERSION_10` 裸信号有 standalone 正收益但方向不稳。
只在 stress-release gate 下启用，可能过滤掉非反弹阶段的 tracking-error noise。

## 方法/配置

基础信号: `SIGNED_TE_REVERSION_10`

Gate: `(TRANS_VIX_DIVERGENCE >= threshold) OR (recent regime_risk elevated AND regime_risk drops over 5d)`

## 结果

| 指标 | 数值 |
|------|------|
| Best IC Horizon | 20d |
| IC Mean | +0.012962 |
| IC Positive Ratio | 0.036 |
| IC t-stat | 4.300 |
| TE Threshold | 0.50 |
| VIX Divergence Threshold | 0.10 |
| Regime Risk Threshold | 0.55 |
| Regime Risk Drop 5d | 0.03 |
| Sharpe | +0.373 |
| Profit Factor | 1.351 |
| Signal Accuracy | 0.590 |
| Num Signals | 210 |
| Max Core Correlation | 0.388 |

## 结论

该 gated signal 仅作为 standalone candidate 留存。若 gate 后仍不能同时通过
IC、profit factor、direction accuracy 和 correlation budget，不进入 RL feature promotion。

## 参考

- Crypto futures basis / funding-rate dislocation analogue
- Stress-release family: compare with `TRANS_VIX_DIVERGENCE`
