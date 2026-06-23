# 中文结论

- 做了什么：用 R-G-E-V-D 流程把 GitHub、论文、Twitter/X、arXiv 外部知识转成 SOXL 候选假设，并通过 qlib.init 与 D.features 读取 SOXL/SOXX/VIX 数据做诊断搜索。
- 核心指标：当前生产基准 qlib_paper_sim_v1 的 test return_pct=6570.0233，max_drawdown_pct=-85.2317；搜索网格共 1350 个路径，最高收益候选 return_pct=7736.8103、max_drawdown_pct=-62.1519，最佳 20% 回撤门槛候选 return_pct=63.4350、max_drawdown_pct=-5.0573。
- regime/window 差异：候选来自趋势、VIX 风险门、目标波动率和高 VIX 去风险组合；高收益候选仍承受深回撤，低回撤候选收益远低于生产基准。
- 是否晋升：不晋升；未找到同时满足收益超过现有策略且 MaxDD < 20% 的候选。
- 风险：这是 diagnostic search，不是 PortAnaRecord 晋升卡；候选路径用 qlib 数据诊断计算，后续若出现过门槛候选必须升级为 qlib recorder/PortAnaRecord validated card。
- 下一步：扩大到模型化 regime selector、intraday T+1 特征和 walk-forward 搜索，而不是继续堆叠静态规则。
- qlib recorder id：5833c5c24a6841929b9aac55bdc7c1d1
