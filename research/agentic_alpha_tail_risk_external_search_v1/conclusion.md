# 结论

- what_ran: 执行 agentic_alpha_tail_risk_external_search_v1，把 Github/论文/Arxiv/Twitter 的 R-G-E-V-D 外部知识映射到 VIX9D、VIX3M、VVIX、SKEW，并用 `build_qlib_store_from_frame` 构建 qlib provider 后通过 `D.features` 审计。
- key_metrics: production baseline return=6570.02%, maxDD=-85.23%; 本卡 passes_both_count=0。
- regime_window_difference: 外部尾部风险数据本地覆盖到 2026-04-17，候选回测有效区间到 2026-04-16，不能冒充完整 2026-06-09 生产窗口。
- promotion_decision: not_promoted，未找到同时满足收益超过现有生产策略且 MaxDD 小于 20% 的候选。
- risks: term-structure 和 vol-target 阈值仍是日频门控，能降低部分尾部风险但会牺牲 SOXL 主要复利窗口；外部数据覆盖也短于生产基准。
- next_action: 转向更长历史的期权尾部风险、实时资金流/订单流或新模型家族，不再重复单纯阈值门控。
