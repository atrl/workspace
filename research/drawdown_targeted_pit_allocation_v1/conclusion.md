# 结论

- what_ran: 成分股与合成2x标签只进入训练辅助头；开发期和outer推理均只使用原有SOXL/SOXX数据，网络直接输出完整仓位，路径预算候选仅根据此前已实现净值与峰值因果缩放风险仓位。
- key_metrics: 开发期选择 `pit_budget_risky_250`，10bp相对SOXX=0.6253，25bp=0.5701，正超额年份=2/7；outer 10bp=0.5276，MaxDD=-29.16%。
- regime_window: SEC filing_date 后使用PIT成员；更早历史并非PIT，只在单独候选中以0.25权重作为预训练样本。
- promotion_decision: `research_target_30pct_met_not_promoted`；相对目标通过=False，完整风险门=False，30%研究目标=True。
- risks: 失败门为 development_20pct_excess_10bps, development_25bps_cost_stress, development_positive_relative_years, outer_20pct_excess_10bps, outer_25bps_cost_stress, outer_three_of_four_positive_years, max_drawdown_minus_20pct, worst_rolling_5d_minus_12pct, upside_capture_60pct；实际单股2x历史太短，只用于校准合成标签，不伪造早期可交易历史。
- next_action: 30%路径风险目标与跑赢SOXX目标分开判定；只有开发期先达到目标后才解释outer诊断，不用outer反选。
- recorder: `4fd1fba883ee49d298e1c6e2d38e8b29`。
