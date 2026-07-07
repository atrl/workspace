# 中文结论

- 做了什么：新增 v5 causal meta-label / leakage audit，用 qlib D.features 读取 SOXL 行上的 close、volume、VIX、SOXX underlying 和 v1 因果特征；LGB/XGB 只预测事件日下一段 open-to-open SOXL 是否为正。
- 核心指标：SOXX benchmark return=413.6565%，max drawdown=-43.0919%；causal candidate 通过数=0。
- 有效边界：回撤安全候选最高收益=106.6544%，收益达标候选中的最浅回撤=-24.4164%。
- 未来泄漏：invalid future-label audit 通过数=2；这类结果直接使用 `etf_open_to_open_return` 或 `underlying_open_to_open_return`，只能作为防未来函数反例，不能晋升。
- 为什么：只要把未来 open-to-open label 当成特征，收益/回撤目标会出现假突破；剔除这些列后，LGB/XGB 对事件方向的分辨率不足以同时保持 SOXX+20 和 -10% 回撤。
- 是否晋升：不晋升；decision=not_promoted_causal_meta_label_goal_failed。
- 下一步：继续优化必须换成开盘前真正可知的信息源，例如盘前/期权/半导体成分股实时扩散/事件日历；仅继续调当前日线事件阈值会倾向过拟合。
- qlib recorder：91e3343d22894064b059659403f186f8
- 不构成买卖建议。
