# 结论

- what_ran: 执行 agentic_alpha_meta_label_probability_gate_search_v1，用 qlib `D.features` 读取 SOXL/SOXX/VIX，按 meta-labeling 思路训练 LogisticRegression 与 HistGradientBoosting 年度 walk-forward probability gate。
- key_metrics: production baseline return=6570.02%, maxDD=-85.23%; raw_passes_both_count=4, sanity_passes_both_count=0。
- regime_window_difference: raw 最佳候选集中在少数高置信 dip 事件，best_drawdown_gate_return=13104.85%, MaxDD=-5.52%, 但 trade_count=3 且 max_exposure=100.00x。
- promotion_decision: not_promoted；裸数值 pass 来自极端暴露和稀疏交易，未通过 max_exposure<=10x 与 trade_count>=20 的 sanity gate。
- risks: 样本太少、暴露过高、meta-label probability 可能过拟合近期 dip rebound；不能作为生产策略或目标仓位。
- next_action: 引入独立 crash-risk / options / order-flow 数据，或把 probability gate 约束在可交易暴露内重新搜索。
