# 中文结论

- 做了什么：在当前 Stage1 confidence-veto 上新增开盘 action-value 执行路由。Stage1 继续用独立 shadow 状态产生原目标；执行层只对不超过5% L1 turnover、且不是去风险的交易判断 execute/hold，所有 beta 下降和10:31恢复日都强制执行。
- 标签：用当日执行与继续持有的实现收益差，扣除25bp当日及下一日复位成本；Logistic 按年度扩展训练，2023-2026模型冻结在2022年末。
- 开发期10bp：baseline return=424.9560%、candidate=426.4930%；MaxDD从 -26.7281% 改善到 -26.7234%，worst5保持 -10.7010%。
- 成本压力：candidate 25bp return=335.3517%，仍高于 baseline 25bp=332.7105%。正式对比见 backtest_benchmark_comparison.csv。
- 频率：开盘交易日减少 11.69%，turnover减少 1.63%，10:31恢复语义没有被固定带宽误删。
- 年度稳定性：开发期逐年财富下限门=True；完整年度比值见 annual_metrics.csv。
- outer复用诊断：baseline=380.2061%、candidate=388.8195%；不参与选择，也不是clean holdout。
- 决策：validated_research_candidate_for_clean_forward_shadow。这已经超过当前研究基线，但生产替换仍 blocked，必须先做独立shadow状态账本和新的连续前向验证。
- qlib：DatasetH/DataHandlerLP 承载年度扩展训练，预测、标签、训练帧、模型和portfolio replay均进入 recorder 77f2ae6c0ea143c0a765366abae71aaf。
- 不构成买卖建议。
