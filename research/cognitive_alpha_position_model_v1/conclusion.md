# 结论

- what_ran: 执行 cognitive_alpha_position_model_v1，先通过 src.factor 的 CogAlpha factor mining 挖掘 alpha library，再把 selected factors 接入 LightGBMClassifier 仓位模型，并用 qlib recorder 记录 checkpoint 和 positions。
- data: provider=/Users/rziiiii/dev/workspace/data/qlib/soxl，as_of=2026-06-24，train=2019-01-02 到 2024-12-31，validation=2025-01-02 到 2025-12-31，test=2026-01-02 到 2026-06-24。
- key_metrics: selected=lgbm_classifier_probability_p0.45_s1_h0.2_hold3，test_return=27.5111%，MaxDD=-12.2054%，latest_target=0.00%。
- direct_factor_backtest: selected=mined_canary_tsmom_signed_budget_h03_direct，return=576.1573%，MaxDD=-14.6197%，beats_constrained=True。
- benchmark: constrained_weight_reference=571.2201%，soxl_backtest=137.9962%，buyhold=389.8814%。
- promotion_decision: candidate_for_validated_research_direct_factor。direct_factor_backtest 已超过 constrained_weight_reference；LightGBM 仓位模型仍保留为诊断，不单独晋升。
- risks: CogAlpha 论文核心是 factor mining；本卡的仓位层只是下游诊断，SOXL 单标的 IC/仓位模型仍可能过拟合；本研究面向交易复盘和风险控制，不构成买卖建议。
- goal_mode: goal_manifest.json 已记录 data/evolution/position/direct_factor_backtest/card/signal 的完整 stage manifest，可作为下一轮 resume/audit 入口。
- next_action: 进入 validated research 复核：做 walk-forward、月度窗口、最新数据刷新和 paper-sim；qlib recorder id=1123b25ef2fb445fa9667f02f992b79c。
