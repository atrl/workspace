# 结论

- what_ran: 执行 agentic_alpha_session_effect_search_v1，用 qlib `D.features` 读取 SOXL/SOXX OHLC/VIX，测试 overnight momentum、intraday gap reversal、tug-of-war close-to-close 三条 session-effect 分支。
- key_metrics: production baseline return=6570.02%, maxDD=-85.23%; passes_both_count=0。
- regime_window_difference: 最强收益分支是 overnight_momentum，best_return=539007.35%, MaxDD=-97.26%；回撤门控版本仍远低于生产基准。
- promotion_decision: not_promoted，未找到同时满足收益超过现有生产策略且 MaxDD 小于 20% 的候选。
- risks: overnight/intraday 分解解释了部分路径差异，但单独使用 OHLC session effect 无法压住 SOXL 杠杆路径回撤；高收益版本仍遭遇深度回撤。
- next_action: 需要新的独立信息源，例如期权曲面、订单流、或更长 intraday demand/supply imbalance；不要重复纯 session split 搜索。
