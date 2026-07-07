# 中文结论

- 做了什么：新增 v3 信号容量诊断，先用 hindsight oracle 检查 SOXX/SOXL/cash 资产集合上界，再用 LGB/XGB crash-risk classifier 测试 base、横向 3x ETF context、完整 lagged 15m 因子。
- 核心指标：SOXX benchmark return=413.6565%，max drawdown=-43.0919%；capacity parameter-oracle 通过数=0。
- 上界：hindsight avoid_worst_80 return=14585.9976%，max drawdown=-8.4749%，说明目标需要强坏日识别，不是资产集合完全不可达。
- 因子：cross_asset_context_feature_count=75，intraday15m_feature_count=17，lag_days=1。
- 风险边界：回撤安全候选最高收益=58.0310%，收益达标候选中的最浅回撤=-23.1993%。
- 是否晋升：不晋升；decision=not_promoted_signal_capacity_goal_failed。15m 因子改善 tail rank，但仍不足以满足 SOXX+20 与 -10% 回撤。
- 下一步：若目标不放宽，需要接入更接近事前冲击的信息，例如期权/波动率曲面、盘前流动性、新闻/宏观事件日历或更高频实时状态；继续只调日线/lagged 15m 阈值意义有限。
- qlib recorder：a556eb3d0e74436180b9c5e731b74550
- 不构成买卖建议。
