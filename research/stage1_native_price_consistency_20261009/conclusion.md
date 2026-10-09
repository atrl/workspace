# 同源价格与原生账户验证准备（中文结论）

通过 Qlib Data API 读取真实日线和 15 分钟 OHLCV，构建有来源哈希的价格视图。
执行视图逐值保留原始 OHLCV，开盘与首小时收盘用真实观察价；日线前缀保持早期来源，自实际日内数据起点起使用日内真实 OHLCV 聚合，避免跨频账户边界价格不一致。

原始日线开盘价与新视图差异：[{'instrument': 'SOXL', 'sessions': 2698, 'maximum_relative_open_change': 0.013624906539916992, 'median_relative_open_change': 1.537799835205078e-05, 'over_one_bp_sessions': 4}, {'instrument': 'SOXX', 'sessions': 2698, 'maximum_relative_open_change': 0.0006535649299621582, 'median_relative_open_change': 1.2814998626708984e-05, 'over_one_bp_sessions': 611}]。

决定：不晋升。本卡仅证明价格来源和视图读回，不证明策略收益、控制器正确性或真实成交。下一步用冻结动作与费用一致候选分别重放，在 Qlib 原生账户中核现金、股数、费用和逐日结算。
