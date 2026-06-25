# 结论

- what_ran: 执行 cognitive_alpha_position_model_v1，把 CogAlpha full evolution 选出的 alpha 接入 LightGBMClassifier 仓位模型，并用 qlib recorder 记录 evolution checkpoint 和 positions。
- data: provider=/Users/rziiiii/dev/workspace/data/qlib/soxl，as_of=2026-06-24，train=2019-01-02 到 2024-12-31，validation=2025-01-02 到 2025-12-31，test=2026-01-02 到 2026-06-24。
- key_metrics: selected=lgbm_classifier_binary_p0.55_s1_h0.25_hold3，test_return=287.1802%，MaxDD=-23.5484%，latest_target=0.00%。
- benchmark: constrained_weight_reference=571.2201%，soxl_backtest=137.9962%，buyhold=389.8814%。
- promotion_decision: not_promoted_under_constrained_weight_reference。未超过 constrained_weight_reference 时不进入 production。
- risks: 这是 full profile evolution 驱动的仓位诊断；SOXL 单标的 IC/仓位模型仍可能过拟合；本研究面向交易复盘和风险控制，不构成买卖建议。
- goal_mode: goal_manifest.json 已记录 data/evolution/position/card/signal 的完整 stage manifest，可作为下一轮 resume/audit 入口。
- next_action: 继续接入真实外部 LLM provider 后再比较是否进入 validated research。qlib recorder id=c2111a3efc4e422aafcd4cf4ffc9104b。
