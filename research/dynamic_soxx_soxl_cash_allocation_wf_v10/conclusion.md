# 中文结论

- 为什么换方案：纯 ML 仓位在 walk-forward 上不稳定；v10 改为复用 v2 已记录的 LGB/XGB regime template，再加固定风险预算。
- 做了什么：动态配置 SOXX + SOXL + cash；selected policy=risk_budget_scale0p40_soxxfloor0p10，SOXL scale=0.4，SOXX floor=0.1。
- 核心指标：SOXX benchmark return=413.6565%，maxDD=-43.0919%；selected return=458.1614%，maxDD=-8.2249%，excess=44.5048%。
- 源 template：raw return=5364.8779%，raw maxDD=-19.8208%；v10 不是直接上 raw template，而是用 0.4 风险预算把回撤压到 -10% 内。
- 泛化判断：通过当前 SOXX+20pp / maxDD>-10% 研究门，但 overfit 风险不是零；缓解点是 v10 参数固定、source template 记录 selection_uses_test_segment=false。
- 是否晋升：candidate_for_validated_research_soxx_excess_goal_passed；可进入 paper/sim 观察，但不能直接实盘。
- qlib recorder：d61a39aa4fbc43c0b4933f844d04d83d
- 不构成买卖建议。
