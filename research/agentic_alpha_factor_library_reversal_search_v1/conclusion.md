# 结论

- what_ran: 执行 agentic_alpha_factor_library_reversal_search_v1，从 Github/论文/Arxiv/Twitter 检索到的 R-G-E-V-D 思路映射到本地 factor library 的 reversal、sell_score 和 volatility-managed 分支，并通过 qlib `D.features` 读取 SOXL/SOXX 日线重新评估。
- key_metrics: production baseline return=6570.02%, maxDD=-85.23%; 本卡 candidate_rows=1224, passes_both_count=0。
- regime_window_difference: 特征暖身使用 2010-03-11 到 2026-06-10，production-window 候选覆盖 2019-01-02 到 2026-06-09。
- promotion_decision: not_promoted，未找到同时满足收益超过现有生产策略且 MaxDD 小于 20% 的候选。
- risks: 因子库强 IC 信号在组合层面仍然被 SOXL 路径风险主导；回撤 gate 内的参数主要靠降暴露，而不是发现了新的高收益低回撤结构。
- next_action: 下一轮不要继续重复 RSI/BIAS/reversal 阈值，优先转向更长历史的期权风险、资金流、订单流或非阈值化模型族。
