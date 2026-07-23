# 显式先验 SOXX 超额策略结论

- 跑了什么：从 hierarchical_soxx_excess_walk_forward_v1 的 qlib recorder 读取年度 OOS 三头预测，把旧 base_constrained_ref 作为独立加法先验，而不是模型特征；仅用 2016–2022 选择五个预定义 profile。
- 开发期：10bp 终值/SOXX=1.2841，25bp=1.1331；MaxDD=-46.98% vs SOXX -47.37%；最长水下 252 vs 257 天。
- Outer 诊断：10bp 终值/SOXX=1.6142；MaxDD=-42.26% vs -42.79%；最长水下 247 vs 297 天；回撤恢复 63 vs 113 天。
- 下跌区间现金：outer 平均 15.63%，最低 10.00%。
- 决定：validated_research_goal_passed_explicit_prior_overlay；这是研究目标通过，不等同于未观察数据上的生产晋升。
- 风险：增量同时依赖模型收益头与显式规则先验；必须持续分别报告 model-only、prior-only 和 blend，不能重新把先验塞回特征矩阵。
- 下一步：进入 paper target-position 观察期，只允许新到数据做前向验证，不再用 2023+ 历史选择参数。
