# 中文结论

- 做了什么：实现 daily open2open 主仓位 + lagged 15m regime 因子 + SOXX 成分股特征的 LGB/XGB hybrid 年度 walk-forward 卡。
- 核心指标：latest test return_pct=108.8446，max_drawdown_pct=-16.2856，annual target pass years=[2024]。
- regime/window 差异：15m 数据只作为 T+1 日级特征和风险 cap，不使用同日未完成 bar，不做独立日内总仓位。
- 是否晋升：not_promoted_hybrid_intraday15m_walk_forward_goal_failed。
- 风险：SOXX 成分股仍带有持仓快照偏差；15m 因子是 bar 聚合特征，不含 bid/ask 队列成交验证；不构成买卖建议。
- 下一步：若年度目标仍失败，优先检查失败年份的买入保持率和 15m risk gate 是否过度削减趋势暴露。
- qlib recorder：7a9956e458f244dd829e1e68bf3b0745
