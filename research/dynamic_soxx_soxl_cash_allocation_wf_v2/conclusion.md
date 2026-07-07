# 中文结论

- 做了什么：重跑 SOXX/SOXL/cash 动态配置 v2 边界诊断，包含 direct-action LGB/XGB、base_soxx_diffusion 成分股特征探测，以及趋势/动量规则上限探测。
- 核心指标：SOXX benchmark stitched return=413.6565%，max drawdown=-43.0919%；满足 SOXX+20 且回撤不低于 -10% 的候选数=0。
- 风险边界：回撤安全候选的最高收益=117.8117%，收益达标候选中的最浅回撤=-25.7722%。
- 日内数据：15min execution sleeve 状态=ok，test_return=13.753271102905273，test_dd=-8.642077445983887，role=candidate_intraday_execution_sleeve_not_core_allocation_alpha。
- 是否晋升：不晋升；decision=not_promoted_dynamic_allocation_frontier_goal_failed。当前证据说明目标缺口来自收益/回撤不可同时满足，而不是单一参数未调。
- 下一步：若继续追求该目标，需要新增可提前识别半导体拥挤去杠杆的外部/实时风险数据，或放宽回撤目标；继续调日线仓位模板意义有限。
- qlib recorder：07d4bb5d601d4e0f85ab1298f5acaae0
- 不构成买卖建议。
