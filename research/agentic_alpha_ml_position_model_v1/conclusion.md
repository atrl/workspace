# 结论

- what_ran: 执行 agentic_alpha_ml_position_model_v1，用 qlib.init + D.features 构建 SOXL/SOXX 日线特征、agentic component exposure 和旧 ensemble raw target，再用 LogisticRegression、HistGradientBoosting、LGBM、Ridge 系列模型训练仓位信号。
- key_metrics: validation 选中 hist_gradient_boosting_regressor_base_gate_old_raw_cap1_p0.55_s1_h0.1_hold1；2026 test return=17.1388%，MaxDD=-1.1839%，avg_abs_position=0.0234，trade_count=14。
- benchmark: constrained_weight_reference return=571.2201%，soxl_backtest return=132.0780%，oracle test-top ML return=45.4258%；ML validation-selected 和 oracle test-top 都没有超过 soxl_backtest，更没有超过 constrained weight。
- regime/window differences: train=2019-2024，validation=2025，只用 validation 选模型和阈值；2026-01-02 到 2026-06-09 是留出报告。validation 通过资格门槛的 ML 候选在 2026 明显低仓位，说明它更像风险过滤器，不是可替代的收益引擎。
- promotion_decision: 不晋升。当前 ML 架构能降低回撤和仓位，但收益与交易覆盖不足，不能替代约束权重模型，也不能作为 SOXL production target-position。
- risks: 单标的日线样本少；特征来自已挖掘 agentic components；LGBM/Ridge 等模型仍可能对 2025 regime 过拟合；真实成交还要考虑跳空、滑点和 paper-sim drift。
- next_action: 如果继续 ML 路线，下一步应改成 walk-forward/online refit + explicit exposure floor 或以 soxl_backtest 风格的仓位预算作为先验，而不是直接用 validation 静态模型替代仓位管理。
