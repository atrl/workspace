# 结论

- what_ran: 修正训练、平滑与回测的换手口径，统一按实际盘前持仓计 SOXL/SOXX 真实订单，并以 qlib DatasetH 年度 walk-forward 比较 4 个预注册候选。
- key_metrics: `pit_holdings_event_0005` 裸模型开发期交易日=602/1762，减少 46.68%，终值为同批基线 113.80%；但 MaxDD=-57.69%。
- controller_integration: 接回冻结 Stage1 后开发期交易日减少 35.70%，但终值仅保留 38.27%，MaxDD=-39.61%。
- regime_window: 2016–2022 用于选择；2023–2026 只做复用诊断。
- promotion_decision: `not_promoted_pit_auxiliary_training_gates_failed`；生产 Stage1、Futu SIMULATE 和当前调度均不切换。
- risks: 裸模型风险过高，Stage1 集成后的收益保留与风险门失败；outer 不用于调参。
- next_action: 保留冻结预测器，单独训练 `HOLD / REBALANCE / DE-RISK` 动作层；降风险立即，加风险需新信号与冷却确认。
- recorder: `77a5c2bb26f34fe0bf6e19d45f15a2a5`。
