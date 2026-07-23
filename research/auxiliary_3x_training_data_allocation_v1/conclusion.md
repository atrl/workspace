# 结论

- 已完成“其他3x只作为训练数据”的严格对照；测试集始终只有SOXL。
- 开发期选择：`tech_spxl_aux_weighted`；10bp相对SOXX=0.9786，25bp=0.8629，正超额年份=6/7，MaxDD=-52.54%。
- Outer仅报告：10bp相对SOXX=1.1578，MaxDD=-45.94%。
- 晋升：`not_promoted_auxiliary_training_data_risk_gates_failed`。
- 失败门：development_return_10bps, development_return_25bps, development_all_risk_gates, outer_year_robustness, outer_maxdd。
- 解释：辅助行改善了部分年度稳定性，但共同归一化特征丢失了SOXL专属alpha，且换手成本抵消了收益；不能替代原有特征模型。
- 下一步：保留该结果作为数据增强边界，不继续扩大pooled权重；若再用其他3x，只允许训练辅助风险头并要求明显超过SOXL-only对照。
- Qlib recorder：`d0d1e39b74fc4203ac28a25386edff3a`。
