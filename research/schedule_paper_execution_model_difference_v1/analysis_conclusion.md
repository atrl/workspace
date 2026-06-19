# 中文结论

- 做了什么：验证 `src/schedule` 的 paper execution pipeline 从 dashboard export 和 canonical data registry 运行到 paper adapter，并在通知、payload、ledger 中保留模型差异解释。
- 核心指标：top model `random_forest_clf__all_features` 原始仓位来自 dashboard 的 `avg_position=0.74`，因 `block_or_rework_loser` 被降到 `target_position=0.1`，paper 结果持有 8 股。
- regime 差异：`stress_window::stress` 中 winner 为 `ridge_mse__all_features`，loser 为 `random_forest_clf__all_features`，spread 为 `6.5`。
- 是否晋升：这是 schedule/API paper/sim 审计链路验证，可随框架进入 production；不是策略晋升。
- 风险：本实验使用最小 fixture 数据和 mock dashboard，不证明真实模型表现。
- 下一步：真实 production notebook 生成 `model_difference_explanations` 后，paper execution 通知会自动暴露该降仓原因。