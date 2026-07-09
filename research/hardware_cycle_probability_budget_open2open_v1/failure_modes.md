# Failure Modes

- promotion_decision: not_promoted_failed_walk_forward_risk_or_return_gates
- failed_checks: ["walk_forward_risk_gates", "direct_base_comparison"]
- architecture_replacement_pass: True
- walk_forward_risk_gate_pass: False
- walk_forward_return_gate_pass: True
- selected_vs_base_constrained_ref_return_gap_pct: -275.15771394988434
- primary_limitation: 该替代结构通过年度 walk-forward 风险/收益门，但收益显著低于 base_constrained_ref。
- implementation_limit: min_position 是固定小仓位 floor，不来自 base_constrained_ref；若用于实盘仍需单独 paper 验证。
- constituent_limit: SOXX 成分股数据仍受 point-in-time 覆盖限制。
