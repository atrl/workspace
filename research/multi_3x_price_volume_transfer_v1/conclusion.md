# 结论

- 执行内容：用非SOXL的长期3x ETF逐underlying检验8个点时量价压力因子，冻结后在2023至2026年SOXL上零样本揭盲。
- 关键指标：SOXL样本外 price-only AUC 0.705、Q5-Q1 13.1%；price-volume AUC 0.432、Q5-Q1 -6.4%、AP lift 0.86x。
- 保留因子：price_volume_divergence_pressure_20, signed_volume_selling_pressure_20。
- 晋升决定：not_retained_no_transferable_price_volume_evidence；该结论只决定是否进入下一轮风险控制研究，不是交易策略晋升。
- 风险：日频量价、ETF上市期差异、早期少量零成交量记录和20日路径标签可能限制泛化，未评估仓位收益与交易成本。
- 下一步：只对跨组且SOXL增量门槛同时通过的因子做仓位节流实验；其余因子停止调模型。
