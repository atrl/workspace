# 结论

- what_ran: 执行 agentic_alpha_ml_factor_enhanced_throttle_v1，用 qlib.init + D.features 构建 SOXL/SOXX 日线、tail risk、macro credit、intraday T+1、cross-asset rotation 增强特征，并训练 Logistic/Tree/LightGBM/Ridge 系列 ML factor-enhanced throttle。
- key_metrics: validation 选中 lgbm_classifier_base_throttle_canary_base_p0.4-0.65_t0.35-1_s1_h0.25_hold2；2026 test return=188.7222%，MaxDD=-12.0300%，avg_abs_position=0.3233，trade_count=15，next_day_reversal_ratio=0.0000。
- benchmark: previous ML return=17.1388%，previous ML throttle return=182.3375%，soxl_backtest return=132.0780%，constrained_weight_reference return=571.2201%，oracle_return_test_top=286.5589%。
- regime/window differences: train=2019-2024，validation=2025，test=2026-01-02 到 2026-06-09；模型学习 5 日 label，但回放用 daily PnL。新架构把仓位上限和换手惯性放在 policy 中，ML 只缩放 base prior。
- promotion_decision: not_promoted_ml_factor_enhanced_under_constrained_weight。若没有超过 constrained_weight_reference，则只作为 ML overlay 诊断，不替代当前约束权重模型。
- risks: 单标的 ML 样本仍小；tail/cross 因子在 2026-04 后新鲜度不足，缺失值由训练集 imputer 处理；constrained_base 本身已经很强，throttle 可能降低趋势期收益；oracle test-top 仍不能当作生产证据。本研究面向交易复盘和风险控制，不构成买卖建议。
- next_action: 保留为研究卡，后续若继续 ML，应做 rolling walk-forward/online refit 和 production paper-sim。qlib recorder id=c0f1405d4b074499922cf2ca406a7f92。
