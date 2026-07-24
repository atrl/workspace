# 结论

- what_ran: 成分股与合成2x标签只进入训练辅助头；开发期和outer推理均只使用原有SOXL/SOXX数据，网络直接输出完整仓位，路径预算候选仅根据此前已实现净值与峰值因果缩放风险仓位。
- key_metrics: 开发期选择 `pit_only_aux_005`，10bp相对SOXX=1.2295，25bp=1.1162，正超额年份=6/7；outer 10bp=1.3834，MaxDD=-48.50%。
- regime_window: SEC filing_date 后使用PIT成员；更早历史并非PIT，只在单独候选中以0.25权重作为预训练样本。
- promotion_decision: `not_promoted_pit_auxiliary_training_gates_failed`；相对目标通过=True，完整风险门=False，30%研究目标=False。
- risks: 失败门为 max_drawdown_minus_20pct, worst_rolling_5d_minus_12pct, upside_capture_60pct, outer_research_worst_5d_target；实际单股2x历史太短，只用于校准合成标签，不伪造早期可交易历史。
- next_action: 30%路径风险目标与跑赢SOXX目标分开判定；只有开发期先达到目标后才解释outer诊断，不用outer反选。
- recorder: `e3f97e26b5d941bbadbdcb3a578fee8c`。
