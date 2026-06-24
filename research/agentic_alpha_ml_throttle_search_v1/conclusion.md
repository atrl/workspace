# 结论

- what_ran: 执行 agentic_alpha_ml_throttle_search_v1，用 qlib.init + D.features 构建 SOXL/SOXX 日线特征、agentic component exposure、constrained_base/old_raw/canary base prior，并训练 Logistic/Tree/LightGBM/Ridge 系列 ML throttle。
- key_metrics: validation 选中 extra_trees_classifier_base_throttle_constrained_base_p0.4-0.65_t0.75-1_s1_h0.25_hold3；2026 test return=182.3375%，MaxDD=-9.3259%，avg_abs_position=0.3163，trade_count=13，next_day_reversal_ratio=0.0000。
- benchmark: previous ML return=17.1388%，soxl_backtest return=132.0780%，constrained_weight_reference return=571.2201%，oracle_return_test_top=287.8333%。
- regime/window differences: train=2019-2024，validation=2025，test=2026-01-02 到 2026-06-09；模型学习 5 日 label，但回放用 daily PnL。新架构把仓位上限和换手惯性放在 policy 中，ML 只缩放 base prior。
- promotion_decision: not_promoted_ml_throttle_under_constrained_weight。若没有超过 constrained_weight_reference，则只作为 ML overlay 诊断，不替代当前约束权重模型。
- risks: 单标的日线 ML 样本仍小；constrained_base 本身已经很强，throttle 可能降低趋势期收益；oracle test-top 仍不能当作生产证据。
- next_action: 保留为研究卡，后续若继续 ML，应做 rolling walk-forward/online refit 和 production paper-sim。qlib recorder id=3d415e61f8a94d278ab49f7fe2068d16。
