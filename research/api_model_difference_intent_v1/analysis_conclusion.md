# 中文结论

- 做了什么：验证 `src/api` 从 dashboard 的 `model_difference_explanations` 读取模型差异解释，并写入 paper intent ledger。
- 核心指标：top model `random_forest_clf__all_features` 的原始 `avg_position=0.74`，因为在 `stress_window::stress` 被标记为 loser 且 decision 为 `block_or_rework_loser`，最终 `target_position=0.1`。
- regime 差异：stress regime 中 winner 是 `ridge_mse__all_features`，loser 是 `random_forest_clf__all_features`，相对收益 spread 为 `6.5`。
- 是否晋升：这是 API paper/sim 风控上下文能力验证，可随框架进入 production；不代表交易策略晋升。
- 风险：本实验使用 mock dashboard payload，只验证 contract 传递和 paper adapter 行为，不替代真实回测。
- 下一步：真实 dashboard export 进入 paper execution schedule 后，ledger 可用于审计为什么 target position 被压低。