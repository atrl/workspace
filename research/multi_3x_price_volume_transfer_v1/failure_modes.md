# Failure Modes

- decision: not_retained_no_transferable_price_volume_evidence
- failed_gates: outer_auc, outer_ap_lift_increment, outer_quintile_lift_increment, annual_direction_consistency
- factor_risk: 日频成交量无法识别盘中吸筹和派发顺序，拆股或数据源调整也可能改变量能分布。
- data_risk: 早期低流动性产品存在少量零成交量记录；因子将其视为零相对量，且SOXL外层样本没有该问题。
- transfer_risk: 3x ETF 的产品年龄、行业集中度和再平衡摩擦不同，跨标的一致不保证半导体周期有效。
- target_risk: breakdown 标签是波动率缩放的20日路径结果，不等同于可交易收益或择时胜率。
