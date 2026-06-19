# 中文结论

- 做了什么：用最小 dashboard payload 验证 `src/schedule` 在 `notify_details=True` 时读取 `model_difference_explanations`。
- 核心指标：`model_difference_count=1`，top signal 为 `baseline_defends`，decision 为 `block_or_rework_loser`，segment 为 `stress_window::stress`，spread 为 `6.5`。
- regime 差异：在 `stress_window::stress` 区间，`ridge_mse` 相对 `gcn_mse` 是 winner，说明定时通知可以直接暴露区间模型差异。
- 是否晋升：仅为 schedule 通知能力证据，可随本次 `src/schedule` 改动进入 production 框架；不代表策略晋升。
- 风险：本实验使用 mock dashboard export，只验证通知字段传递，不验证真实回测指标质量。
- 下一步：接入真实 dashboard export 后，可在通知系统中按 signal/decision 分流提醒。