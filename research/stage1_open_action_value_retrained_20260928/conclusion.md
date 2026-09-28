> 费用会计复核：本次历史结果已由 `stage1_open_action_value_net_fees_20260928` 的净现金费用回放替代。原结果保留供审计，不能用于激活。

# 中文结论

- 做了什么：在固定 Stage1 confidence-veto 上新增开盘 action-value 执行路由。训练标签来自baseline分支；正式验证每天用候选自身净值、峰值、历史收益和风险状态重新计算目标。执行层只对不超过5% L1 turnover、非去风险且当前持仓合法的交易判断 execute/hold；09:31不读取未来恢复信号，10:31复用当日开盘控制器快照后按首小时信号独立执行。
- 标签：用当日执行与继续持有的实现收益差，扣除25bp当日及下一日复位成本；Logistic 按年度扩展训练，2023-2026模型固定使用2022年末及以前的信号行。标签记录收益结算与次日目标均已可知的时间，只允许在实际测试开盘前已成熟的标签进入训练。
- 开发期10bp：baseline return=115.0401%、candidate=114.4380%；MaxDD为 -29.9305% / -30.1501%，candidate worst5=-10.8237%。
- 成本压力：candidate 25bp return=82.9576%，baseline 25bp=83.1767%。正式对比见 backtest_benchmark_comparison.csv。
- 频率：开盘交易日减少 17.18%，turnover减少 0.99%，10:31恢复语义没有被固定带宽误删。
- 年度稳定性：开发期逐年财富下限门=True；完整年度比值见 annual_metrics.csv。
- outer复用诊断：baseline=220.3700%、candidate=219.7036%；不参与选择，也不是clean holdout。
- 决策：not_promoted_open_action_value_development_gates_failed。开发门全部通过=False；生产晋升仍 blocked，通过开发门后还须新的连续前向模拟验证。
- qlib：DatasetH/DataHandlerLP 承载年度扩展训练，预测、标签、训练帧、模型和portfolio replay均进入 recorder 398436752f7a41a6887a90307a6355f2。
- 不构成买卖建议。
