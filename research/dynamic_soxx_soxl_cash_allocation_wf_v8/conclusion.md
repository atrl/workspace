# 中文结论

- 做了什么：新增 v8 15m intraday stop + CPPI 风险预算边界，并加入 LGB/XGB/平均模型的 risk-budget selector。
- 容量结果：frontier rows=6480，passes=0，best drawdown-safe return=148.9191%，maxDD=-9.4674%。
- ML 结果：selected=avg_xgb_lgb，stitched return=-1.5847%，maxDD=-8.8122%，excess=-415.2412%。
- 结论：15m intraday stop 与 CPPI 能降低回撤，但一旦把回撤压到 10% 内，SOXL 平均风险预算过低，无法达到 SOXX+20；验证期选择的 ML 风险预算也未通过，不晋升。
- 是否晋升：不晋升；decision=not_promoted_intraday_cppi_ml_risk_budget_goal_failed。
- 下一步：如果继续冲击该目标，需要引入更细的 execution alpha 或不同目标约束；当前日线信号加 15m 风控不足以证明实盘能力。
- qlib recorder：2a2993a5ce1f4bbaa6a0c225fbca5dbf
- 不构成买卖建议。
