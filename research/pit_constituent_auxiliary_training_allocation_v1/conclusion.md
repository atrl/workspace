# 结论

- what_ran: 成分股与合成2x标签只进入训练辅助头；开发期和outer推理均只使用原有SOXL/SOXX数据，仓位由一个网络直接输出。
- key_metrics: 开发期选择 `pit_only_aux_005`，10bp相对SOXX=1.2797，25bp=1.1596，正超额年份=6/7；outer 10bp=1.4996，MaxDD=-48.26%。
- regime_window: SEC filing_date 后使用PIT成员；更早历史并非PIT，只在单独候选中以0.25权重作为预训练样本。
- promotion_decision: `not_promoted_pit_auxiliary_training_gates_failed`；相对目标通过=True，完整风险门=False。
- risks: 失败门为 max_drawdown_minus_20pct, worst_rolling_5d_minus_12pct, upside_capture_60pct；实际单股2x历史太短，只用于校准合成标签，不伪造早期可交易历史。
- next_action: 若20%相对门通过但风险门失败，应保留辅助训练头，把研究目标转为可预测的现金风险期限；不再扩大pooled权重。
- recorder: `3144ec23c3ee4478b6b1e699fb8203d3`。
