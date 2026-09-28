# 中文结论

- 做了什么：在固定 Stage1 confidence-veto 上新增开盘 action-value 执行路由。训练标签来自baseline分支；正式验证每天用候选自身净值、峰值、历史收益和风险状态重新计算目标。执行层只对不超过5% L1 turnover、非去风险且当前持仓合法的交易判断 execute/hold；09:31不读取未来恢复信号，10:31复用当日开盘控制器快照后按首小时信号独立执行。
- 标签：用当日执行与继续持有的实现收益差，扣除25bp当日及下一日复位成本；Logistic 按年度扩展训练，2023-2026模型固定使用2022年末及以前的信号行。标签记录收益结算与次日目标均已可知的时间，只允许在实际测试开盘前已成熟的标签进入训练。
- 开发期10bp：baseline return=115.0363%、candidate=170.7450%；MaxDD为 -29.9306% / -30.0832%，candidate worst5=-10.8237%。
- 成本压力：candidate 25bp return=131.8550%，baseline 25bp=83.1615%。正式对比见 backtest_benchmark_comparison.csv。
- 频率：开盘交易日减少 14.94%，turnover减少 3.26%，10:31恢复语义没有被固定带宽误删。
- 年度稳定性：开发期逐年财富下限门=True；完整年度比值见 annual_metrics.csv。
- outer复用诊断：baseline=220.3673%、candidate=219.4260%；不参与选择，也不是clean holdout。
- 决策：not_promoted_open_action_value_development_gates_failed。开发门全部通过=False；生产晋升仍 blocked，通过开发门后还须新的连续前向模拟验证。
- qlib：DatasetH/DataHandlerLP 承载年度扩展训练，预测、标签、训练帧、模型和portfolio replay均进入 recorder b472a458fc9042b4905094fddeaebd33。
- 不构成买卖建议。
