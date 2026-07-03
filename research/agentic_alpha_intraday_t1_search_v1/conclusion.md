# 中文结论

- 做了什么：把 R-G-E-V-D 的下一轮接到 intraday T+1 qlib store，使用 `build_qlib_store_from_frame` 构建 `data/qlib/soxl_intraday_t1`，再通过 `D.features`、`DatasetH/DataHandlerLP`、`R.start` 和 `LGBMClassifier`/`XGBClassifier` 跑 intraday_t1_lgbm_replay、intraday_t1_xgb_replay 与 intraday_t1_overlay 候选分支。
- 核心指标：生产基准 qlib_paper_sim_v1 full-test return_pct=6570.0233、max_drawdown_pct=-85.2317；本卡 intraday 分支 top-level passes_both_count=0。
- 分支结果：intraday_t1_lgbm_replay 最高收益候选 return_pct=215.5740、max_drawdown_pct=-8.7541，最佳 20% 回撤候选 return_pct=215.5740、max_drawdown_pct=-8.7541；intraday_t1_xgb_replay 最高收益候选 return_pct=224.3112、max_drawdown_pct=-15.7332，最佳 20% 回撤候选 return_pct=224.3112、max_drawdown_pct=-15.7332；intraday_t1_overlay 最高收益候选 return_pct=2668.3304、max_drawdown_pct=-71.6418，最佳 20% 回撤候选 return_pct=306.9492、max_drawdown_pct=-16.8835。
- regime/window 差异：intraday T+1 sidecar 覆盖 2019-01-02 到 2026-07-02，LGBM 分支 test 从 2025-01-02 开始，overlay 分支因滚动波动率从 2019-02-01 开始；这些窗口只能做诊断，不能独立证明 2019-2026 全窗口晋升。
- 是否晋升：不晋升；未找到同时满足收益超过现有生产策略且 MaxDD < 20% 的候选。
- 风险：intraday 已覆盖 2019-01-02 至 2026-07-02，但当前只是 T+1 日频安全特征；LGBM/XGB 搜索器仍未晋升，下一步若做开盘成交/盘中风控，需要 extended-hours 或全日分钟执行回放，并用外部源交叉校验。
- qlib recorder：1f9c0f86731c4ad6954d58dbb57f7446
