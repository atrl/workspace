# 分层 SOXL/SOXX/cash walk-forward 结论

- 跑了什么：2016–2022 年度 expanding walk-forward，三类特征包、两类模型、三个独立预测头；2023–2026-07-02 仅作 outer 诊断。
- 开发期：10bp 终值比 SOXX=1.1404，25bp=0.9583；MaxDD=-36.27% vs SOXX -47.37%。
- Outer：10bp 终值比 SOXX=0.9197；MaxDD=-41.58% vs SOXX -42.79%。
- Regime/窗口：开发期最长水下 205 天 vs SOXX 257 天；downside 平均现金 23.28%。
- 晋升决定：needs_iteration_hierarchical_soxx_excess_transfer。
- 风险：策略超额依赖收益头对少数上行窗口的识别；现金动作样本稀疏，outer 若未转移则不能因历史开发期通过而晋升。
- 下一步：若未通过 outer，按 head_oos_audit、outer_head_transfer_diagnostic、action_attribution 和 policy_ablation 区分收益头失效、hazard 误杀或 cash gate 召回不足，不再扩大无约束参数网格。
