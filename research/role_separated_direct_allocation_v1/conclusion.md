# 结论

- 已执行三头cross-fit与直接仓位分配；选择完全不使用outer。
- 开发期诊断候选：`crossfit_alpha_recovery_direct_mse`；10bp相对SOXX=1.0405，25bp=1.0039，MaxDD=-53.77%。
- Outer仅报告：相对SOXX=1.0346，MaxDD=-47.01%。
- 晋升：`not_promoted_role_or_risk_gates_failed`。
- 主要失败门：development_all_gates, outer_year_robustness, outer_drawdown, hazard_learnability。
- 风险：头部可学性不足时，router不能创造新信息；outer不得反向影响模型选择。
- 下一步：根据head_summary判断继续改目标，还是停止网络搜索并补充长期PIT期权/高频数据。
- Qlib recorder：`3020e2ceb0274c35ad205835862c89cf`。
