# 中文结论

- 做了什么：建立 delayed observable intraday overlay，日线 LGB/XGB open-to-open 仓位仍决定是否交易；日内层必须等前 30/60 分钟完成后，才能用 observed probability/return 条件触发后续 15m next-open 入场。
- 选择规则：只用 validation；要求 20bp 和 50bp validation 同时满足 85%-95% retention、max_dd>-18%、skip<=15%；test 只报告。
- 候选：4736 个 policy、14208 行成本压力结果；validation+test 同时过门槛行数=38。
- validation 选中：obs30_observed_prob_mean_ge_q0.525_0.464215_signal_0.70，20bp retention=87.48%，max_dd=-14.29%；50bp retention=88.96%，max_dd=-17.12%。
- test 报告：20bp retention=85.79%，max_dd=-11.20%；50bp retention=88.29%，max_dd=-12.36%。
- 是否晋升：not_promoted_external_intraday_history_insufficient。短窗口内这个 delayed entry 方向确实把 test max_dd 从 baseline 约 -18.38% 压到更低，但外部 15m 历史和真实成交 replay 还不足，不能直接晋升生产。
- qlib recorder：b6ffad6f927743228e083bf957e0058f
