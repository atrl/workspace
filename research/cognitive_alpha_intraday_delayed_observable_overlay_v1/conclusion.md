# 中文结论

- 做了什么：建立 delayed observable intraday overlay，日线 LGB/XGB open-to-open 仓位仍决定是否交易；日内层必须等前 30/60 分钟完成后，才能用 observed probability/return 条件触发后续 15m next-open 入场；本轮把 same_day_close 作为可晋升的纯日内平仓候选加入，同时保留 hold_to_next_open；risk-exit 仍只作为诊断候选，避免短样本风险退出过拟合。
- 选择规则：只用 validation；retention 必须落在 85%-95% hard gate 内，再按 validation skip rate、条件覆盖率和回撤改善排序；test 只报告，不参与阈值选择。超过 95% 的短窗口收益保留为诊断，不计入晋级目标。
- 候选：28416 个 policy、85248 行成本压力结果；validation+test 同时过门槛行数=120；其中 same_day_close 同时过门槛行数=6，risk-exit 同时过门槛行数=48。
- validation 选中：obs30_observed_prob_mean_ge_q0.400_0.420242_signal_0.45_hold，exit_policy=hold_to_next_open；20bp retention=94.95%，max_dd=-13.73%；50bp retention=95.00%，max_dd=-14.28%。
- test 报告：20bp retention=99.16%，max_dd=-15.31%；50bp retention=99.97%，max_dd=-16.05%。
- 日内 flat sleeve：只在 same_day_close 候选内按 validation 回撤优先选择；选中 obs60_observed_prob_min_ge_q0.525_0.229488_signal_0.60_same_day_close；20bp validation retention=88.23%，max_dd=-10.55%，test retention=40.34%，test max_dd=-18.38%；50bp test retention=38.69%，test max_dd=-19.00%。这个 sleeve 更接近纯日内执行，但仍受外部 15m 历史不足和收盘成交 replay 缺失约束。
- walk-forward：要求 [2023, 2024, 2025, 2026] 年度 train_to_T-2/valid_T-1/test_T；当前 feasible 年份=[]，缺失=[2023, 2024, 2025, 2026]，因此 required_policy_satisfied=False。
- 是否晋升：not_promoted_delayed_overlay_test_gate_failed。短窗口内这个 delayed entry 方向确实把 test max_dd 从 baseline 约 -18.38% 压到更低，但外部 15m 历史和真实成交 replay 还不足，不能直接晋升生产。
- qlib recorder：82c4fc7f4eb442d89da9df4c7c19b2b9
