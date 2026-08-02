# 结论

- what_ran: 冻结 63 因子基线的模型家族和仓位规则，对 18 个宏观特征逐个 add-one，并从全量宏观模型逐个 drop-one。
- period: train=['2019-01-02', '2024-12-31']，validation=['2025-01-02', '2025-12-31']，outer_test=['2026-01-02', '2026-07-30']，另含 2023/2024/2025/2026YTD 四个 walk-forward 测试折。
- baseline: validation_score=42.528345，test_return=7.1073%，maxDD=-4.0717%。
- best_add_one_by_validation: macro_dfii10_z252，validation_score_delta=12.170005，test_return_delta=2.7872%，paired_CI=[-4.0870%, 11.1710%]。
- best_conditional_drop_one: macro_dtwexbgs_z252，validation_score_contribution=11.579672，test_return_contribution=-2.3914%。
- strict_gate: qualified=[]。
- promotion_decision: not_promoted_no_strict_incremental_feature。
- risks: 18 次探索存在多重检验问题；2026YTD 配对区间样本较短；模型权重会随特征集重训，因此结论是固定架构/仓位下的预测增量，不是因果效应。
- next_action: 只有 strict gate 通过的列才进入独立 validated research；若为空，保持当前 63 因子基线不变。
