# 中文结论

- 做了什么：冻结现有 Stage1、Stage2 关闭和 30% SOXL 上限，使用 qlib DatasetH 将 10:30 后合法 beta 网格的最优增量蒸馏为连续 Ridge 目标；开发期逐年 walk-forward，2023–2026 共用截至 2022 年的冻结候选模型；对照组保留现有 Stage1 的逐年更新语义。
- 开发期：候选 `ridge_a100_s025_u015` 10bp return=461.3196%，相对 Stage1 增量=44.0475 个百分点，MaxDD=-26.5279%，worst-5d=-11.8194%。
- 换手：开发期 turnover reduction=1.99%，要求=20.00%，gate=False。
- outer 复用诊断：return=514.1861%，相对 Stage1 增量=-27.6029 个百分点；该区间不参与选择，也不是 clean holdout。
- 决策：`not_promoted_turnover_or_robustness_gate_failed`；生产 Stage1 不变，真实资金晋升 blocked。
- 风险：失败门=['turnover_gate_pass']；连续模型明显增加动作频率，历史开发期改善不足以替代新的连续前向模拟。
- 下一步：保持本卡为失败边界证据；若继续，只能预登记更强的换手/动作稀疏惩罚并等待 clean-forward，不用 outer 调阈值。
- qlib recorder：`c4c5082fa9eb4be3a6789deb7b80538a`。
- 不构成买卖建议。
