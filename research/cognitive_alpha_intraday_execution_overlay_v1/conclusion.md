# 中文结论

- 做了什么：建立“日级模型 + 日内执行 overlay”实验，日级 LGB/XGB 仓位模型仍决定整体 target_position，15m lgb_xgb_consensus 只负责下一交易日的入场过滤、前 4 根 15m TWAP 和盘中风险退出。
- 选择规则：只用 validation、20bp 成本选择 variant；test 只报告，没有用 test 调阈值或调参数。
- validation 选择：baseline_next_open，retention=100.00%，max_dd=-18.14%。
- test 对照：baseline_next_open return=353.82%，selected return=353.82%、max_dd=-18.38%。
- open-buy-dump 诊断：-3% 当日入场后回撤阈值，baseline test 次数=8，selected test 次数=8，减少=0；crash 次数减少=0。
- 条件化 risk sleeve：validation 只用 probability le 0.6280 时切到 overlay_b_twap4；validation open-buy-dump 减少=4，test retention=85.38%，test max_dd=-16.63%，test open-buy-dump 减少=3。
- walk-forward：要求 [2023, 2024, 2025, 2026] 年度 train_to_T-2/valid_T-1/test_T；当前 feasible 年份=[2023, 2024, 2025, 2026]，缺失=[]，required_policy_satisfied=False。
- 最新年度 fold：wf_2026 validation 选择 overlay_a_signal_gate，test retention=13.54%，test max_dd=-0.01%，passes_target=False。
- 是否晋升：not_promoted_overlay_did_not_beat_daily_execution。selected 仍是 baseline，当前 15m 执行层没有可靠超过 next-open 一次成交。
- 风险：这不是独立日内 alpha；它依赖日级模型的趋势和隔夜收益来源。真实推广前还要补 bid/ask replay、成交队列、延迟和盘前盘后风险验证。
- qlib recorder：f4900bbceadc4080a5d5de3416eecdc7
