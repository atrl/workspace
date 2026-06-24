# 结论

- what_ran: 执行 agentic_alpha_constrained_weight_search_v1，用 qlib.init + D.features 重建 agentic ensemble 分支暴露，并搜索 long-only、max_position<=100%、多因子有效分支数>=2 的权重预算。
- key_metrics: selected=canary_tsmom_budget_h025，YTD return=571.2201%，MaxDD=-14.6197%，max_position=1.00，effective_components=2。
- benchmark: raw target-position reference return=479.1566%，MaxDD=-18.6394%；soxl_backtest return=132.0780%，MaxDD=-13.5541%。
- promotion_decision: promote_to_production_review；本卡超过 raw target-position reference、超过 soxl_backtest，且 MaxDD 小于 20%，同时仓位不超过 100%。
- regime/window differences: canary trend 是主仓位信号，TSMOM 低波动分支提供辅助暴露；交易次数=9，next_day_reversal=1。
- latest_signal: latest as_of=2026-06-23，selected_policy_target=0.0375%；latest 只读仓位，不计入验证收益。
- risks: 该候选仍是 YTD constrained search，存在样本内选择风险；需要下一步 production paper/sim 和 walk-forward 验证。qlib recorder id=98332c7ea723467e85411e962c57ba31。
- next_action: 将该权重作为 production review 候选，继续做 walk-forward、月度窗口和最新 macro/data refresh 后的 paper-sim。
