# Failure Modes

- promotion_decision: not_promoted_under_base_constrained_ref
- failed_checks: []
- selected_factor_count: 1
- candidate_roles: {"daily_cloud_gpu_proxy_promotable": 1}
- cloud_daily_proxy_role: 云 GPU 主特征改为日频公开市场代理；真实租赁价格月频状态只作为诊断，不作为主晋升信号。
- free_direct_price_role: 免费日频 GPU 租赁报价已接入，但历史覆盖短且缺口不前填，默认只作为诊断候选。
- promotion_rule: 只有 position model 和 walk-forward gates 同时通过，才可进入 validated research；单个硬件因子 IC 不作为晋升依据。
