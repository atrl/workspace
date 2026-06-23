# 中文结论

- 做了什么：把 R-G-E-V-D 的下一轮接到 intraday T+1 qlib store，使用 `build_qlib_store_from_frame` 构建 `data/qlib/soxl_intraday_t1`，再通过 `D.features`、`DatasetH/DataHandlerLP`、`R.start` 和 `LGBMClassifier` 跑 intraday_t1_lgbm_replay 与 intraday_t1_overlay 两个候选分支。
- 核心指标：生产基准 qlib_paper_sim_v1 full-test return_pct=6570.0233、max_drawdown_pct=-85.2317；本卡 intraday 分支 top-level passes_both_count=0。
- 分支结果：intraday_t1_lgbm_replay 最高收益候选 return_pct=565.3522、max_drawdown_pct=-58.2925，最佳 20% 回撤候选 return_pct=214.7394、max_drawdown_pct=-19.1761；intraday_t1_overlay 最高收益候选 return_pct=399.9297、max_drawdown_pct=-61.6272，最佳 20% 回撤候选 return_pct=76.3477、max_drawdown_pct=-19.2761。
- regime/window 差异：intraday T+1 sidecar 覆盖 2024-01-03 到 2026-06-05，LGBM 分支 test 从 2025-01-02 开始，overlay 分支因滚动波动率从 2024-02-02 开始；这些窗口只能做诊断，不能独立证明 2019-2026 全窗口晋升。
- 是否晋升：不晋升；未找到同时满足收益超过现有生产策略且 MaxDD < 20% 的候选。
- 风险：intraday 历史只有 2024 以后，样本短；LGBM 只作为搜索器，不是生产固定参数；下一步应补更长 intraday 历史或把通过回撤门槛的 intraday gate 融入已有日线 validated/paper-sim 链路继续比较。
- qlib recorder：99f2dc48d9f543829ac9fd871a4f5f91
