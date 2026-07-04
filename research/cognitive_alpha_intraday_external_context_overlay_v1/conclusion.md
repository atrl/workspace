# 中文结论

- 做了什么：建立“外部 15m 上下文执行 overlay”实验，日级 LGB/XGB 仓位模型仍决定整体 target_position，外部 15m 上下文 triple-barrier/execution-improvement 模型只负责下一交易日的入场过滤、前 3/4 根 15m TWAP 和盘中风险退出。
- 选择规则：只用 validation、20bp 成本选择 variant；test 只报告，没有用 test 调阈值或调参数。
- validation 选择：overlay_a_signal_gate，retention=93.00%，max_dd=-16.79%。
- 阈值搜索：只用 validation 搜 960 个候选，满足 85%-95% retention 且 max_dd>-18% 的候选数=24。
- test 对照：baseline_next_open return=353.82%，selected return=271.80%、max_dd=-15.02%。
- open-buy-dump 诊断：-3% 当日入场后回撤阈值，baseline test 次数=8，selected test 次数=4，减少=4；crash 次数减少=2。
- 条件化 risk sleeve：validation 只用 probability le 0.6202 时切到 overlay_b_twap3；validation open-buy-dump 减少=2，test retention=88.42%，test max_dd=-15.45%，test open-buy-dump 减少=2。
- walk-forward：要求 [2023, 2024, 2025, 2026] 年度 train_to_T-2/valid_T-1/test_T；当前 feasible 年份=[2026]，缺失=[2023, 2024, 2025]，required_policy_satisfied=False。
- 最新年度 fold：wf_2026 validation 选择 overlay_a_signal_gate，test retention=76.82%，test max_dd=-15.02%，passes_target=False。
- 是否晋升：not_promoted_external_intraday_history_insufficient。selected test retention=76.82% 未过 85%-95% 目标，说明 validation-only 选择仍不能推广。
- 风险：这不是独立日内 alpha；它依赖日级模型的趋势和隔夜收益来源。真实推广前还要补 bid/ask replay、成交队列、延迟和盘前盘后风险验证。
- qlib recorder：4c69544f9c3b4282a180a2ba94ca3d0a
