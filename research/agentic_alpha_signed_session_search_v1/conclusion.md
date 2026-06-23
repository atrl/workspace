# 中文结论

- 做了什么：继续 R-G-E-V-D，把 time-series momentum、overnight/intraday reversal 和 Tug-of-War 的外部知识转成 signed session long/short 候选，并通过 qlib `D.features` 读取 SOXL/SOXX OHLC/VIX 后调用 `src.factor.evaluate_signed_session_effect_candidates`。
- 核心指标：生产基准 qlib_paper_sim_v1 full-test return_pct=6570.0233、max_drawdown_pct=-85.2317；本卡 passes_both_count=0。
- 分支结果：signed_overnight_momentum 最高收益 return_pct=230037.2597、max_drawdown_pct=-94.4088，最佳 20% 回撤候选 return_pct=451.1520、max_drawdown_pct=-15.6807；signed_intraday_momentum 最佳 20% 回撤候选 return_pct=40.9092；signed_tug_of_war_close_to_close 最佳 20% 回撤候选 return_pct=129.2798。
- regime/window 差异：overnight signed momentum 是唯一有明显收益弹性的 session 分支，但高收益路径仍在 2020/2022/2025-2026 压力段承受深回撤；20% DD 内的 0.5x signed overnight 只能保留约数百个百分点收益，无法超过生产 baseline。
- 是否晋升：not_promoted；未找到同时满足收益超过现有生产策略且 MaxDD >= -20% 的候选。
- 风险：这是 diagnostic search，不是生产门；signed session 单独使用无法达成目标，后续只能作为新模型的候选特征或风险解释，不应直接提升为 production strategy。
- qlib recorder：ec61dbaee29d4737861261e112a98314
