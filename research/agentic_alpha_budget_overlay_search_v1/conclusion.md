# 中文结论

- 做了什么：继续 R-G-E-V-D，把 Github/论文/Arxiv/Twitter 的 LETF 趋势、vol-target、CPPI/TIPP 风险预算思路叠到现有 production recorder 的 `pred.pkl` 和 `label.pkl`，并通过 qlib `D.features` 读 SOXL/SOXX/VIX 状态后跑 hard gate 与 TIPP budget 两个分支。
- 核心指标：生产基准 qlib_paper_sim_v1 full-test return_pct=6570.0233、max_drawdown_pct=-85.2317；本卡 passes_both_count=0。
- 分支结果：production_signal_budget_overlay 最高收益候选 return_pct=16107.3898、max_drawdown_pct=-94.8658，最佳 20% 回撤候选 return_pct=310.8535、max_drawdown_pct=-19.3817；tipp_budget_overlay 最佳 20% 回撤候选 return_pct=269.0667、max_drawdown_pct=-18.8160。
- regime/window 差异：高收益路径集中在允许高暴露的趋势窗口，但这些路径在 2020/2022/2025-2026 的 SOXL 压力段出现深回撤；CPPI/TIPP 能压住回撤，但会把均值暴露压低到无法超过 production baseline。
- 是否晋升：不晋升；未找到同时满足收益超过现有生产策略且 MaxDD < 20% 的候选。
- 风险：这是 diagnostic search，不是生产门；它关闭的是“同一 production score 上继续叠 budget overlay”的分支，下一步应换预测模型族、补更长 intraday 历史，或研究明确的 crash-risk alpha。
- qlib recorder：d0d0260966724a83b4dbcc9a92f1c921
