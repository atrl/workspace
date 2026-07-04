# 中文结论

- 做了什么：建立 causal regime overlay 诊断卡，日线 LGB/XGB open-to-open 仓位仍是 baseline，外部 15m LGB 模型只作为执行 overlay；条件选择只允许使用 daily close 时已知的 probability、target_position、turnover、raw_target。
- 选择规则：只用 validation、20bp 成本和 validation quantile 阈值选条件；test 只报告，不参与调参。
- 候选数：666；validation 过 85%-95% retention 且 max_dd>-18% 的候选数=34；validation 和 test 同时过门槛的候选数=16。
- validation 选中：overlay_a_signal_gate / raw_target_le_valid_q0.800_1，retention=94.85%，max_dd=-8.32%。
- test 报告：retention=13.22%，max_dd=-2.33%。
- 是否晋升：not_promoted_regime_overlay_test_gate_failed。结论是当前 causal 日内 overlay 还不能替代日线 next-open 执行；下一步应优化日线整体仓位模型或做真正延迟入场/风险退出模型，而不是用看完第一小时后的特征证明开盘 TWAP。
- qlib recorder：27d8e59e0a9642a1b8697c241f0a8583
