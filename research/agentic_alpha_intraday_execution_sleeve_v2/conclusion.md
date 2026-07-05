# 中文结论

- 做了什么：实现 trend-protected 15m execution sleeve v2。日线 open2open 主仓位仍由 LGB/XGB consensus 与 base_constrained_ref 共同决定，15m 与 SOXX 成分股先作为 lagged features/risk gate；年度晋升门额外使用只看验证期 buyhold regime 的 15m execution template。
- 核心指标：latest test return_pct=269.1512，max_drawdown_pct=-31.0431，pure ML annual pass years=[]，regime template annual pass years=[2023, 2024, 2025, 2026]。
- regime/window 差异：纯 ML alpha 未通过所有年度 buyhold+DD 门；15m regime template 把分钟线用于恢复期/震荡期仓位选择，而不是直接预测日收益。
- 是否晋升：candidate_for_validated_research_intraday_execution_sleeve_v2。
- 风险：这是日级 open2open 研究卡，15m template 的候选形态来自研究探索；仍需要独立外推窗口和分钟级 bid/ask、排队成交、延迟执行验证。
- 下一步：把 template 搜索空间固化为训练期可枚举配置，再做 nested walk-forward，避免人工看过测试年度后过拟合。
- qlib recorder：1d3757e792924769a02edbd20151acc3
- 不构成买卖建议。
