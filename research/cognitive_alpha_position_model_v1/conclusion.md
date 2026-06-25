# 结论

- what_ran: 执行 cognitive_alpha_position_model_v1，把 CogAlpha smoke evolution 选出的 alpha 接入 LightGBMClassifier 仓位模型，并用 qlib recorder 记录 evolution checkpoint 和 positions。
- data: provider=/Users/rziiiii/dev/workspace/data/qlib/soxl，as_of=2026-06-23，train=2019-01-02 到 2024-12-31，validation=2025-01-02 到 2025-12-31，test=2026-01-02 到 2026-06-23。
- key_metrics: selected=lgbm_classifier_probability_p0.45_s1_h0.2_hold3，test_return=67.7975%，MaxDD=-10.1946%，latest_target=29.44%。
- benchmark: constrained_weight_reference=571.2201%，soxl_backtest=137.9962%，buyhold=389.8814%。
- promotion_decision: not_promoted_under_constrained_weight_reference。未超过 constrained_weight_reference 时不进入 production。
- risks: 这是 smoke evolution，不是论文级 full profile；SOXL 单标的 IC/仓位模型仍可能过拟合；本研究面向交易复盘和风险控制，不构成买卖建议。
- next_action: 跑 soxl_full profile 后再比较是否进入 validated research。qlib recorder id=1750df2fda8d465383a804dcd2502cf9。
