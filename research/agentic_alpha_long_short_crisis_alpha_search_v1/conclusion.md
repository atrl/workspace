# 结论

- what_ran: 执行 agentic_alpha_long_short_crisis_alpha_search_v1，把 Time Series Momentum、trend-following 和 crisis-alpha 外部知识映射到 SOXL/SOXX/VIX 的 long-short trend 与 crisis overlay 分支，并通过 qlib `D.features` 读取日线重新评估。
- key_metrics: production baseline return=6570.02%, maxDD=-85.23%; 本卡 candidate_rows=1440, passes_both_count=0。
- regime_window_difference: 特征暖身使用 2010-03-11 到 2026-06-10，production-window 候选覆盖 2019-01-02 到 2026-06-09。
- promotion_decision: not_promoted，未找到同时满足收益超过现有生产策略且 MaxDD 小于 20% 的候选。
- risks: 单资产 SOXL 的 trend-following/short crisis sleeve 没有复制 diversified futures 的 crisis alpha；做空或对冲腿增加换手和反转损失，仍无法压住路径回撤。
- next_action: 下一轮不要继续扩展单资产 trend-following 阈值，优先尝试真正正交的数据或模型族。
