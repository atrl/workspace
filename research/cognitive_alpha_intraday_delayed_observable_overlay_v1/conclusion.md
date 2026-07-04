# 中文结论

- 做了什么：建立 delayed observable intraday overlay，日线 LGB/XGB open-to-open 仓位仍决定是否交易；日内层必须等前 30/60 分钟完成后，才能用 observed probability/return 条件触发后续 15m next-open 入场；默认 selected 仍持到 following open，不是纯日内平仓；本轮还把 risk-exit 作为诊断候选加入，但默认 selected 只允许 hold_to_next_open，避免短样本风险退出过拟合。
- 选择规则：只用 validation；retention 用 85% floor，95% 只作为 scoring soft cap，再按 validation skip rate、条件覆盖率和回撤改善排序；test 只报告，不参与阈值选择。
- 候选：23680 个 policy、71040 行成本压力结果；validation+test 同时过门槛行数=4007；其中 risk-exit 同时过门槛行数=2605。
- validation 选中：obs30_observed_return_le_q0.550_0.00175977_signal_0.45_hold，exit_policy=hold_to_next_open；20bp retention=277.33%，max_dd=-14.24%；50bp retention=322.43%，max_dd=-15.31%。
- test 报告：20bp retention=188.86%，max_dd=-9.14%；50bp retention=195.78%，max_dd=-9.44%。
- walk-forward：要求 [2023, 2024, 2025, 2026] 年度 train_to_T-2/valid_T-1/test_T；当前 feasible 年份=[]，缺失=[2023, 2024, 2025, 2026]，因此 required_policy_satisfied=False。
- 是否晋升：not_promoted_external_intraday_history_insufficient。短窗口内这个 delayed entry 方向确实把 test max_dd 从 baseline 约 -18.38% 压到更低，但外部 15m 历史和真实成交 replay 还不足，不能直接晋升生产。
- qlib recorder：d9eecb07213a4518943d480d394f5fc8
