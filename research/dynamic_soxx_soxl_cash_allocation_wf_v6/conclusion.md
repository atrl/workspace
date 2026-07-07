# 中文结论

- 做了什么：新增 v6 same-day 15m + XGB 阈值校准实验；same-day 15m 指收盘后已经完整可知的当日 15 分钟聚合，用于 next-open 信号，不用于盘前信号。
- 容量结果：capacity pass count=0；收益达标侧 best return=433.8208%，maxDD=-10.2410%；回撤安全侧 best return=410.7294%，maxDD=-9.9279%。
- nested 结果：rolling validation window=4 年，stitched return=236.0008%，maxDD=-11.3379%，excess=-177.6557%。
- 为什么不晋升：正式 replay 下 capacity 已经是 near-miss；nested 阈值迁移后收益和回撤更不能同时过门槛。
- 是否晋升：不晋升；decision=not_promoted_same_day_15m_capacity_and_nested_calibration_failed。
- 下一步：优化方向应集中在在线阈值校准、score 分布漂移修正和按年度/波动 regime 的校准器，而不是继续扩大同一组 XGB 特征。
- qlib recorder：a8f05e338d7d4ed8a558ec737f5bfd24
- 不构成买卖建议。
