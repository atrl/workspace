# 中文结论

- 做了什么：建立 delayed observable intraday overlay，日线 LGB/XGB open-to-open 仓位仍决定是否交易；日内层必须等前 30/60 分钟完成后，才能用 observed probability/return 条件触发后续 15m next-open 入场；本轮还把 risk-exit 作为诊断候选加入，但默认 selected 只允许 hold_to_next_open，避免短样本风险退出过拟合。
- 选择规则：只用 validation；要求 20bp 和 50bp validation 同时满足 85%-95% retention、max_dd>-18%、skip<=15%；test 只报告。
- 候选：23680 个 policy、71040 行成本压力结果；validation+test 同时过门槛行数=114；其中 risk-exit 同时过门槛行数=48。
- validation 选中：obs30_observed_prob_mean_ge_q0.300_0.367315_signal_0.60_hold，exit_policy=hold_to_next_open；20bp retention=92.90%，max_dd=-11.76%；50bp retention=94.51%，max_dd=-12.32%。
- test 报告：20bp retention=72.60%，max_dd=-15.31%；50bp retention=73.72%，max_dd=-16.05%。
- walk-forward：要求 [2023, 2024, 2025, 2026] 年度 train_to_T-2/valid_T-1/test_T；当前 feasible 年份=[]，缺失=[2023, 2024, 2025, 2026]，因此 required_policy_satisfied=False。
- 是否晋升：not_promoted_delayed_overlay_test_gate_failed。短窗口内这个 delayed entry 方向确实把 test max_dd 从 baseline 约 -18.38% 压到更低，但外部 15m 历史和真实成交 replay 还不足，不能直接晋升生产。
- qlib recorder：ac0934bc17e446efa97ab0db744b93f4
