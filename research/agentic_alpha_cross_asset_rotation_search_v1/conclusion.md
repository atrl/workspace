# 结论

- what_ran: 执行 agentic_alpha_cross_asset_rotation_search_v1，把 Github/论文/Arxiv/Twitter 的跨资产动量和 TAA 思路映射到 SOXL/TQQQ/TECL/SPXL/UPRO，构建 qlib provider 后通过 `D.features` 读取并测试 relative momentum rotation 与 cross-asset breadth SOXL 两条分支。
- key_metrics: production baseline return=6570.02%, maxDD=-85.23%; 本卡 passes_both_count=0。
- regime_window_difference: 多资产 raw symbol 共同覆盖到 2026-04-15，候选有效回测到 2026-04-14，不能冒充完整 2026-06-09 生产窗口。
- promotion_decision: not_promoted，未找到同时满足收益超过现有生产策略且 MaxDD 小于 20% 的候选。
- risks: 轮动宇宙仍全是高相关 3x 股指/行业 ETF，风险预算降低回撤时同步削弱复利，不能提供真正独立的崩盘对冲。
- next_action: 转向更长历史的独立对冲资产、期权风险曲面或新模型家族，不再重复同类 leveraged-equity rotation。
