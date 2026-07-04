# 中文结论

- 做了什么：建立 causal regime overlay 诊断卡，日线 LGB/XGB open-to-open 仓位仍是 baseline，外部 15m LGB 模型只作为执行 overlay；条件选择只允许使用 daily close 时已知的 probability、target_position、turnover、raw_target。
- 选择规则：只用 validation、20bp 成本和 validation quantile 阈值选条件；test 只报告，不参与调参。
- 候选数：666；validation 过 85%-95% retention 且 max_dd>-18% 的候选数=122；validation 和 test 同时过门槛的候选数=17。
- validation 选中：overlay_b_twap4 / probability_le_valid_q0.775_0.627992，retention=88.90%，max_dd=-14.34%。
- test 报告：retention=85.38%，max_dd=-16.63%。
- 是否晋升：not_promoted_external_intraday_history_insufficient。结论是当前 causal 日内 overlay 还不能替代日线 next-open 执行；下一步应优化日线整体仓位模型或做真正延迟入场/风险退出模型，而不是用看完第一小时后的特征证明开盘 TWAP。
- qlib recorder：2a4dd86d49d042c3bb18bc59e6f76a50
