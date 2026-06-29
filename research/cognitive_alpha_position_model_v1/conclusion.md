# 结论

- what_ran: 执行 cognitive_alpha_position_model_v1，先通过 src.factor 的 CogAlpha factor mining 挖掘 alpha library，再把 mined factors、Alpha158、canary/TSMOM、VIX 与 realized volatility 接入 LightGBM/Logistic/Ridge/HistGBDT 仓位模型，并用 qlib recorder 记录 checkpoint 和 positions。
- data: provider=/Users/rziiiii/dev/workspace/data/qlib/soxl，as_of=2026-06-24，train=2019-01-02 到 2024-12-31，validation=2025-01-02 到 2025-12-31，test=2026-01-02 到 2026-06-24。
- key_metrics: selected=ridge_probability_p0.45_s1_h0.2_hold3，test_return=0.0000%，MaxDD=0.0000%，latest_target=0.00%。
- direct_factor_backtest: selected=mined_canary_tsmom_signed_budget_h03_direct，return=576.1573%，MaxDD=-14.6197%，beats_constrained=True，promotion=diagnostic_direct_factor_candidate。
- walk_forward: folds=3，positive_ratio=0.3333，monthly_positive_ratio=0.3333，zero_position_ratio=0.8456，max_extreme_loss_days=1，mean_test_return=5.0171%，worst_MaxDD=-7.6406%，all_gates_pass=False，test_exposure_gate=0.05/0.90。
- benchmark: constrained_weight_reference=571.2201%，soxl_backtest=137.9962%，buyhold=389.8814%。
- promotion_decision: not_promoted_failed_position_risk_gates。direct_factor_backtest 只作为 benchmark/prior；最终晋升仅看 position_model 或 production paper/sim；production_paper_sim=blocked_not_promoted。
- risks: CogAlpha 论文核心是 factor mining；本卡的仓位层只是下游诊断，SOXL 单标的 IC/仓位模型仍可能过拟合；本研究面向交易复盘和风险控制，不构成买卖建议。
- goal_mode: goal_manifest.json 已记录 data/evolution/position/direct_factor_backtest/card/signal 的完整 stage manifest，可作为下一轮 resume/audit 入口。
- next_action: 当前未通过 validated research gate，不创建 production paper/sim；下一步继续做月度窗口、最新数据刷新和模型特征稳定性筛选；qlib recorder id=0a8210c16351487ab25873b1504baaa6。
