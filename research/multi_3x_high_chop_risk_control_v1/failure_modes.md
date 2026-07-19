# Failure Modes

- promotion_decision: not_promoted_risk_control_gates_failed
- failed_gates: upside_capture, average_risky_weight, return_to_drawdown_benchmarks, transfer_fold_pass_rate
- model_risk: 跨ETF可迁移不等于SOXL结构稳定，年度和underlying holdout结果必须同时审阅。
- execution_risk: 仅验证次日开盘持有与固定单边换手成本，未包含滑点冲击。
- data_risk: ETF起始日期不同，晚成立产品的早期fold会按类别样本门槛阻断。
