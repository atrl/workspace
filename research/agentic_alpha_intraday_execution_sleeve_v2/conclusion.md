# 中文结论

- 做了什么：实现 ML-confirmed trend-protected 15m execution sleeve v2。日线 open2open 主仓位由 LGB/XGB consensus 与 base_constrained_ref 共同决定；年度晋升门使用只看验证期 buyhold regime 的 15m execution template，并要求 source daily LGB/XGB probability >= 0.35 后才允许非 source 模板出仓位。
- 核心指标：ML-confirmed annual WF=2023: return=316.3%, DD=-11.3%, B&H=169.1%；2024: return=43.7%, DD=-7.5%, B&H=6.5%；2025: return=81.4%, DD=-19.8%, B&H=77.8%；2026: return=403.5%, DD=-18.0%, B&H=335.4%；目标通过年份=[2023, 2024, 2025, 2026]。纯 position-model latest test return_pct=269.1512，max_drawdown_pct=-31.0431，pure ML annual pass years=[]。
- regime/window 差异：纯 ML alpha 未通过所有年度 buyhold+DD 门；ML-confirmed 15m regime template 把分钟线用于恢复期/震荡期仓位选择，而不是直接预测日收益。
- 是否晋升：candidate_for_validated_research_intraday_execution_sleeve_v2。
- 风险：这是日级 open2open 研究卡，template 的 regime 映射仍来自研究探索；仍需要独立外推窗口和分钟级 bid/ask、排队成交、延迟执行验证。
- 下一步：把 regime 映射扩到跨资产/跨周期训练期枚举，继续降低人工模板过拟合风险。
- qlib recorder：eacc0ba56546431aadb7430a3ef52399
- 不构成买卖建议。
