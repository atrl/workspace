# 中文结论

- 做了什么：冻结当前 Stage1，只在其 SOXL 已触及 30% 时测试额外 5%/10% overflow；资金只从 SOXX 调拨，现金权重不变，信号使用 prior-close 特征并在 next-open 生效。
- 过拟合控制：只搜索 8 个预先固定候选；Logistic C 固定为 0.10，特征固定为 6 个；2016–2017 仅训练，2018–2022 expanding walk-forward 选模，2023–2026 只诊断且所有 outer 年份共用训练截止 2022-12-30 的冻结模型。
- 选中诊断候选：always_005；开发期增量收益=-0.8286 个百分点，动作=69 次，正增量年份=2/5。
- 整段稳健性检查：若把 2016–2017 训练期也并入整段，增量收益=-1.5069 个百分点；该数字同样为负，说明当前 Stage1 口径下并不存在被训练期掩盖的稳定 overflow 收益。
- 开发期风险：MaxDD 变化=-0.4230 个百分点，最差 5 日变化=0.0000 个百分点，正动作比例=57.97%。
- 外层复用诊断：增量收益=25.9956 个百分点，动作=176 次；该结果不得反向选择参数。
- overfit 风险=high；失败项=['minimum_events_per_feature', 'development_all_gates', 'parameter_neighborhood_retention', 'outer_cap_event_rate_ratio', 'clean_unseen_holdout']。主要问题是开发期没有正增量、早期 fold 每个特征对应事件数不足、outer 触顶频率相对开发期发生明显漂移，而且没有干净 holdout。
- 决策：not_promoted_development_negative_high_overfit_risk；不修改生产 Stage1，不开启 overflow paper/sim。只有新增独立 PIT 信息或预登记 clean-forward 后，才值得重开。
- qlib recorder：33b59b75c3c040179c8e341f33ceb50f
- 不构成买卖建议。
