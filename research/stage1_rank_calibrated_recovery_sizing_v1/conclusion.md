# 中文结论

- 做了什么：冻结 Stage1 的 Logistic C=0.03 与最高 0.6 beta，只把硬阈值替换为训练折内经验分位和固定线性恢复；候选数=1，没有参数网格。
- 数据与训练：通过 qlib DatasetH/DataHandlerLP/StaticDataLoader 构建年度扩展训练段，当前预测只与本折训练概率分布比较，不使用测试期排名。
- 开发期 10bp：baseline return=415.4288%，MaxDD=-27.5285%，turnover=149.5013；challenger return=414.1568%，MaxDD=-27.8218%，turnover=143.8927。
- 稳定性：challenger 风险窗口通过=7/7，相对 SOXX 正窗口=7/7，最差相对财富比=1.0022。
- 外层边界：2023–2026 仅为已复用诊断，不参与结构、参数或晋级选择；不能作为干净 holdout。
- 决策：not_promoted_rank_calibrated_recovery_development_gates_failed；production_change=blocked。即使开发门通过，也只能进入冻结 forward shadow，不能替换当前生产 Stage1。
- 风险：固定分位映射仍是一种结构假设；若开发门失败，本路径停止，不调整分位点后重试。
- qlib recorder：f88af79c1cdc45988acbef61d06b0834
- 不构成买卖建议。
