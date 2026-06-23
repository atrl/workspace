# 结论

- what_ran: 执行 agentic_alpha_ml_pair_search_v1，用 qlib `D.features` 读取 SOXL/SOXX/VIX，测试 walk-forward crash/return ensemble 与 SOXL/SOXX pair hedge 两条 R-G-E-V-D 分支。
- key_metrics: production baseline return=6570.02%, maxDD=-85.23%; 新分支 passes_both_count=0。
- regime_window_difference: 日线模型和 pair hedge 都能在部分低波趋势窗口提高收益或降低风险，但不能同时保持高收益和 MaxDD<20%。
- promotion_decision: not_promoted，未找到同时满足收益超过现有生产策略且 MaxDD 小于 20% 的候选。
- risks: 阈值搜索仍依赖日线 SOXL/SOXX/VIX，缺少真正独立的 crash-risk 信息；pair hedge 的 SOXX 对冲会同步削弱 SOXL 牛市收益。
- next_action: 转向新的外部 crash-risk 数据源、期权/波动曲面或更长 intraday/order-flow 特征，不再重复日线 SOXL/SOXX/VIX 门控。
