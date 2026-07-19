# 结论

- 执行内容：不用预测模型，分别检验价格路径耗竭、杠杆脆弱性、固定五龙头结构代理和SOXL/SOXX盘中路径。
- 基线：2023至2026 SOXL高位样本 price-only AUC 0.705、AP lift 1.75x、Q5-Q1 13.1%。
- price_path_exhaustion: selected=none; decision=not_retained_no_incremental_factor_evidence
- leverage_fragility: selected=none; decision=not_retained_no_incremental_factor_evidence
- fixed_bellwether_structure: selected=none; decision=not_retained_no_incremental_factor_evidence
- soxl_soxx_intraday_path: selected=intraday_range_ratio_pressure; AUC=0.401; AP lift=0.92x; Q5-Q1=-6.4%; conditional=-3.9%; decision=not_retained_no_incremental_factor_evidence
- 晋升决定：只有同时改善基线AUC、AP lift、Q5-Q1并在price-only分层内仍有正增量的家族才保留。
- 风险：固定五龙头是代理组合而非历史成分股；外部龙头15分钟历史从2025起，已阻断其长期筛选；2025/2026事件较少时年度值仅作诊断。
- 下一步：只把通过增量门槛的因子接入降仓/波动控制；失败家族停止继续调模型。
