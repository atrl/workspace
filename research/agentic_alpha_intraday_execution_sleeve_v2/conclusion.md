# 中文结论

- 做了什么：保留 trend-protected 15m execution sleeve v2 的纯 position-model 路径；日线 open2open 主仓位由 LGB/XGB consensus 与 base_constrained_ref 共同决定，已移除验证期 regime 映射仓位链路。
- 核心指标：目标通过年份=[]。纯 position-model latest test return_pct=269.1512，max_drawdown_pct=-31.0431，pure ML annual pass years=[]。
- regime/window 差异：纯 ML alpha 未通过所有年度 buyhold+DD 门；分钟线只作为 lagged risk/execution context，不再生成二次规则仓位。
- 是否晋升：not_promoted_intraday_execution_sleeve_v2_annual_goal_failed。
- 风险：这是日级 open2open 研究卡，仍需要独立外推窗口和分钟级 bid/ask、排队成交、延迟执行验证。
- 下一步：只保留可由模型/风险门直接解释的 sizing 结构，继续检查过拟合来源。
- qlib recorder：1627903a9e05438e80236ce6e6f892df
- 不构成买卖建议。
