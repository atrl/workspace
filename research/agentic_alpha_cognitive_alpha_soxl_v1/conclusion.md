# 结论

- what_ran: 执行 agentic_alpha_cognitive_alpha_soxl_v1，把 arXiv:2511.18850 的 CogAlpha 思路落到 `src.factor.cognitive_alpha`：Codex 可生成候选、质量闸门、IC/ICIR/RankIC/RankICIR/MI 筛选；再结合 Alpha158=166 个候选和 seed=8 个 OHLCV 候选，用 qlib SOXL 数据训练 ML throttle。
- key_metrics: validation 选中 extra_trees_classifier_base_throttle_constrained_base_p0.35-0.6_t0.9-1_s1_h0.25_hold3；2026 test return=373.1605%，MaxDD=-21.7470%，avg_abs_position=0.4454，max_position=0.9482，trade_count=18，next_day_reversal_ratio=0.0000。
- factor_mining: selected_alpha_features=21，alpha158_selected=21，cognitive_seed_selected=0，selection_scope=train_plus_validation_only。
- benchmark: factor_enhanced_throttle return=188.7222%，soxl_backtest return=137.9962%，constrained_weight_reference return=571.2201%，oracle_return_test_top=380.2427%。
- promotion_decision: not_promoted_cognitive_alpha_under_constrained_weight。没有超过 constrained_weight_reference 时，只作为架构和研究验证卡，不替代当前主仓位模型。
- risks: CogAlpha 这里是确定性 Codex-compatible 框架，不是完整多代自动进化；SOXL 单标的 IC 是时间序列近似，弱于论文的横截面股票池证据；本研究面向交易复盘和风险控制，不构成买卖建议。
- next_action: 若后续继续，应把 Codex 候选生成接成多代候选池，并用 walk-forward 重新选择阈值。qlib recorder id=2413d524bbba49acb5f2d53549a4befe。
