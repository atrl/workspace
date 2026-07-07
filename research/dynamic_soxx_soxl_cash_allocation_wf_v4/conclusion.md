# 中文结论

- 做了什么：新增 v4 full-factor frontier 诊断，先用 hindsight oracle 检查 SOXX/SOXL/cash 资产集合上界，再用 LGB/XGB action classifier 测试 base、横向 context、lagged 15m、刷新后的 VIX/VVIX/SKEW tail-risk、SOXX 成分股扩散和全量组合特征。
- 核心指标：SOXX benchmark return=413.6565%，max drawdown=-43.0919%；capacity parameter-oracle 通过数=0。
- 上界：hindsight avoid_worst_80 return=14585.9976%，max drawdown=-8.4749%，说明目标需要强坏日识别，不是资产集合完全不可达。
- 因子：cross_asset_context_feature_count=75，intraday15m_feature_count=17，tail_risk_feature_count=18，constituent_feature_count=41，lag_days=1。
- 风险边界：回撤安全候选最高收益=127.4562%，收益达标候选中的最浅回撤=-26.8476%。
- 为什么：SOXX+20 且 maxDD 控制在 -10%以上的约束要求模型长期参与半导体强趋势，同时提前躲开极少数坏日簇；现有因果特征的 tail/action 分辨率不够，放大收益会穿回撤，压回撤会牺牲收益。
- 是否晋升：不晋升；decision=not_promoted_full_factor_frontier_goal_failed。全量因子仍不足以满足 SOXX+20 与 -10% 回撤。
- 下一步：若目标不放宽，需要接入更接近事前冲击的信息，例如期权/波动率曲面、盘前流动性、新闻/宏观事件日历或更高频实时状态；继续只调日线/lagged 15m 阈值意义有限。
- qlib recorder：d9195f5833054eba9e27c1a97a391b46
- 不构成买卖建议。
