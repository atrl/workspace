# Failure Modes

- promotion_decision: not_promoted_under_base_constrained_ref
- failed_checks: []
- selected_factor_count: 2
- candidate_roles: {"proxy_promotable": 2}
- cloud_daily_proxy_role: 云 GPU 主特征改为日频公开市场代理；真实租赁价格月频状态只作为诊断，不作为主晋升信号。
- free_direct_price_role: 免费日频 GPU 租赁报价已接入，但它是云实例小时租赁价，不是 GPU 销售价；历史覆盖短且缺口不前填，默认只作为诊断候选。
- storage_sales_price_role: FRED/BLS 存储设备 PPI 是官方销售价格指数，但为月频；进入模型时只作为带 staleness 的 lagged 状态。
- storage_daily_proxy_role: 硬盘、NAND/SSD、网络互联日频字段来自公开市场供应链股票代理，不是产品级硬盘报价。
- rejected_storage_price_source: PricePerGig 产品级存储价格因 ML 训练和缓存限制不进入 qlib 特征。
- promotion_rule: 只有 position model 和 walk-forward gates 同时通过，才可进入 validated research；单个硬件因子 IC 不作为晋升依据。
