# 中文结论

## 跑了什么

按冻结的 cash-floor benchmark-relative 合同，使用 qlib SOXL/SOXX/SOXS provider、DatasetH/DataHandlerLP 和 recorder，训练两个固定 Ridge 头：五日现金优势与五日未来波动。2016–2022 逐年 expanding walk-forward，20 日 purge；风险资产固定 30% SOXL / 70% SOXX，模型只控制 10%–60% 现金。2023–2026 outer 仅诊断。

## 开发期结果

10bp：收益 268.10%，SOXX 327.07%，相对财富 0.8619，MaxDD -37.39%，最差五日 -14.98%，平均现金 38.52%。25bp 相对财富 0.7336。正相对收益年份 3/7，年度中位财富比 0.9827，bootstrap 跑赢概率 24.71%。

通过门：`['minimum_daily_cash', 'average_cash', 'zero_cash_days', 'maxdd_vs_soxx', 'worst5_vs_soxx', 'maxdd_vs_matched_static', 'worst5_vs_matched_static', 'maxdd_improvement_vs_no_cash', 'worst5_improvement_vs_no_cash']`。失败门：`['relative_wealth_10bps', 'relative_wealth_25bps', 'positive_fold_count', 'median_fold_ratio', 'bootstrap_outperformance_probability', 'timing_value_vs_matched_static']`。

## Outer 诊断

10bp：收益 297.12%，SOXX 392.60%，相对财富 0.8062，MaxDD -40.73%。outer 不参与模型、映射或阈值选择。

## 决定、风险和下一步

`needs_iteration_relative_wealth`。双头 max 映射改善了尾部风险，但平均现金过高且换手成本显著，未达到相对 SOXX 的主目标。不能用 outer 回调；下一步若继续，应在开发期内部重新设计现金聚合函数，例如让波动头限制现金上限或采用平滑/滞后，而不是改变固定风险袖套后宣称通过。

不构成买卖建议。
