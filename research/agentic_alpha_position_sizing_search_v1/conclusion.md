# 结论

- what_ran: 执行 agentic_alpha_position_sizing_search_v1，用 qlib.init + D.features 重建 agentic ensemble 分支暴露，并把 raw signal 转成 SOXL target-position 口径比较。
- key_metrics: selected=cap1_h0.2_chgnone_hold1，YTD return=386.7833%，MaxDD=-19.3887%，max_position=1.00，days_above_100=0。
- benchmark: soxl_backtest YTD return=132.0780%，MaxDD=-13.5541%；raw target-position reference return=479.1566%，MaxDD=-18.6394%。
- regime/window differences: cap + hysteresis 消除了超过 100% 的仓位，next_day_reversal 从 raw 的 11 降到 1，但未超过 raw target-position reference。
- model_view: 浅层概率门控最佳 return=145.8489%，MaxDD=-8.5618%，因此神经网络/模型仓位器当前不应作为第一优先级；更适合先做受约束仓位层。
- latest_signal: latest as_of=2026-06-23，raw_target=28.0325%，selected_policy_target=43.8375%；macro 分支在 2026-06-09 后 forward-filled，只读仓位不验证收益。
- promotion_decision: iterate_not_promoted_until_raw_position_reference_exceeded；该方案超过 soxl_backtest 且 MaxDD 小于 20%，但未超过 raw target-position reference，不能直接晋升。
- risks: 旧卡 8775% 是分支收益合成口径，不等于可执行单目标仓位；macro-credit provider 未更新到 daily provider 最新日期；仓位约束牺牲收益换取更低杠杆和更少隔日反向交易。qlib recorder id=fefeeea750c1494ba819e78642ede6ca。
- next_action: 若要继续追求收益超过 raw reference，需要在仓位预算内重新搜索分支权重，或做 walk-forward 的 constrained optimizer，而不是直接上深度网络。
