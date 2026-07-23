# 多任务分类直接连续配置结论

- 跑了什么：比较 allocation-only、tabular 多任务、快速变化特征多任务和20日GRU多任务；所有候选直接输出连续SOXL/SOXX/cash权重。
- 开发期诊断优胜：tabular_base_multitask；10bp财富/SOXX=0.8745，25bp=0.6082，MaxDD=-43.29%。
- 分类：tail1 AUC=0.5076，relative alpha AUC=0.5968，recovery AUC=0.6275。
- Outer仅诊断：10bp财富/SOXX=0.7809，MaxDD=-54.92%。
- 晋升决定：not_promoted_multitask_classification_or_risk_gates_failed。
- 风险：分类标签若跨年份不可学习，继续更换网络没有意义；outer不得用于选择标签、特征或模型。
- 下一步：只根据开发期分类与风险门决定是继续优化现有特征，还是转向新的PIT盘前/期权/成分股实时信息。
