# 结论

- what_ran: 执行 hardware_cycle_probability_budget_open2open_v1，在纯 probability budget 仓位模型中加入 191 个硬件周期/云 GPU 滞后特征。
- key_metrics: selected_return=2.9563%，maxDD=-0.6529%，max_daily_loss=-0.3171%，CVaR5=-0.2436%。
- walk_forward: folds=4，risk_all_pass=False，return_all_pass=True，min_return=1.4000%。
- hardware_cycle: provider=/Users/rziiiii/dev/workspace/data/qlib/hardware_cycle_v1，calendar=2019-01-02..2026-07-02，lag_days=1。
- latest_signal: as_of=2026-07-02，action=HOLD，target_position=1.5000%。
- benchmark_gap: selected_vs_base_constrained_ref_return_gap_pct=-275.1577%。
- promotion_decision: not_promoted_failed_walk_forward_risk_or_return_gates。
- risks: 云 GPU 直接价格历史较短且是外部重构指数；若收益提升，只能作为研究候选，生产前仍需 point-in-time 数据归档。
- next_action: 对比 probability_budget_open2open_v1 的收益/风险门，确认硬件周期特征是否真实改善 WF。recorder=b717e0064de74cac919121d15ee320f6。
