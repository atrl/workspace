# 中文结论

## 跑了什么

把历史特征按新目标重新整合，而不是复制旧模型 importance。主候选包括当前 10 特征、精简 reintegrated 特征和全量 reintegrated 特征；模型比较 Ridge 与 ExtraTrees。现金头改为未来五日 `60%现金配置相对10%现金配置` 的直接对数财富优势，波动头只作为加权风险限制器，不再逐日取两个头的最大值。2016–2022 expanding walk-forward，20 行 purge；2023–2026-07-02 outer 仅诊断。

## 开发期结果

选中 `current10__ridge__volw_0.75__none`，选择政策 `diagnostic_fallback_relative_gate_minimum_risk_violation`。10bp 收益 364.23%，相对 SOXX 财富 1.0870，MaxDD -42.34%，最差单日 -9.33%，CVaR5 -4.52%，最差五日 -18.38%，平均现金 28.21%。25bp 相对财富 1.0626；正相对收益年份 4/7，年度中位财富比 1.0207。

精简 reintegrated 特征把 Ridge 现金保护头的 pooled Spearman 从 0.0465 提高到 0.0595，但没有转化成更好的组合尾部。最终选择仍是 `current10`，说明旧特征在新标签下只有预测增量，没有稳定的决策增量。选中方案通过了开发期相对财富门，但严格尾部风险门=False。全量特征只作为开发期对照，不因特征更多而优先。

## Outer 诊断

10bp 收益 341.06%，相对 SOXX 财富 0.9143，MaxDD -46.29%。outer 不参与特征、模型、波动权重或 router 选择。

## 决定、风险和下一步

`needs_iteration_tail_risk_after_feature_reintegration`。趋势、距高点、VIX/实现波动、杠杆衰减、tracking error 和跨资产扩散可进入新目标候选，但本轮没有证明它们应取代当前 10 特征。它们仍不能可靠预测足以把 MaxDD 压到 -20% 的尾部时点。SOXX 成分股历史因非 PIT，宏观/硬件/盘中特征因 2019 后才覆盖，均未进入主模型。下一步不应再扩大日频价格特征数量，而应增加真正提前于尾部的 PIT 风险数据或接受风险门与相对收益目标之间的不可行边界。

不构成买卖建议。
