# 中文结论

- 做了什么：继续 R-G-E-V-D 的下一轮，把外部 LETF/vol-control/trend-following 知识转成 model replay、日线 LGBM selector 和跨资产 LGBM selector 三个候选分支，并通过 qlib Data API 与生产 recorder 回放验证。
- 核心指标：生产基准 qlib_paper_sim_v1 full-test return_pct=6570.0233、max_drawdown_pct=-85.2317；本卡三个分支 passes_both_count=0。
- 分支结果：production_signal_overlay 最高收益候选 return_pct=8227.6966 但 max_drawdown_pct=-62.2632；qlib_daily_lgbm_replay 最佳 20% 回撤候选 return_pct=403.3972、max_drawdown_pct=-19.5362；cross_asset_lgbm_replay 只覆盖到 2026-04-10，最佳 20% 回撤候选 return_pct=175.8683。
- 是否晋升：不晋升；未找到同时满足收益超过现有生产策略且 MaxDD < 20% 的候选。
- 风险：这是 diagnostic search，不是 validated PortAnaRecord 晋升卡；跨资产分支覆盖期短于 full-test，不能作为完整目标证据。
- 下一步：需要转向更强的信息面，例如把 2024-2026 Futu 1m intraday T+1 特征接入 qlib store，并做 walk-forward intraday-aware selector，而不是继续扩大日线阈值网格。
- qlib recorder id：d6544d1ee4ec4a54b8e9cc847e5b928f
