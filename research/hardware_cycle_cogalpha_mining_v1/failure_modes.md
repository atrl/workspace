# Failure Modes

- promotion_decision: not_promoted_failed_walk_forward_risk_gates
- failed_checks: ["walk_forward_risk_gates"]
- selected_factor_count: 1
- candidate_roles: {"proxy_promotable": 1}
- cloud_direct_role: 云 GPU 直接价格历史较短，带有 source_observed/is_carried/staleness 语义，默认按诊断特征处理。
- promotion_rule: 只有 position model 和 walk-forward gates 同时通过，才可进入 validated research；单个硬件因子 IC 不作为晋升依据。
