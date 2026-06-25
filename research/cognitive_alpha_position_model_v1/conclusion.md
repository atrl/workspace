# 结论

- what_ran: 执行 cognitive_alpha_position_model_v1，先通过 src.factor 的 CogAlpha factor mining 挖掘 alpha library，再把 selected factors 接入 LightGBMClassifier 仓位模型，并用 qlib recorder 记录 checkpoint 和 positions。
- data: provider=/Users/rziiiii/dev/workspace/data/qlib/soxl，as_of=2026-06-24，train=2019-01-02 到 2024-12-31，validation=2025-01-02 到 2025-12-31，test=2026-01-02 到 2026-06-24。
- key_metrics: selected=lgbm_classifier_binary_p0.55_s1_h0.25_hold3，test_return=287.1802%，MaxDD=-23.5484%，latest_target=0.00%。
- benchmark: constrained_weight_reference=571.2201%，soxl_backtest=137.9962%，buyhold=389.8814%。
- promotion_decision: not_promoted_under_constrained_weight_reference。未超过 constrained_weight_reference 时不进入 production。
- risks: CogAlpha 论文核心是 factor mining；本卡的仓位层只是下游诊断，SOXL 单标的 IC/仓位模型仍可能过拟合；本研究面向交易复盘和风险控制，不构成买卖建议。
- goal_mode: goal_manifest.json 已记录 data/evolution/position/card/signal 的完整 stage manifest，可作为下一轮 resume/audit 入口。
- next_action: 后续直接用 Codex provider 扩展候选池，不再把外部 API key 作为运行前置条件；若要推广，仍需先超过 constrained_weight_reference。qlib recorder id=e1d9e03c29d344d7ba47d8df4970a843。
