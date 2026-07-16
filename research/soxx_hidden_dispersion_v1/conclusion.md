# 结论与反思

- what_ran: 使用 qlib SOXX 成分股 OHLCV、SOXX/SOXL open 与 VIX，构建内部离散度、抵消、相关性破裂、领导轮换和成交量 breadth 特征。
- data: as_of=2026-07-02，constituents=30，PIT=blocked_current_holdings_backfill。
- direction: baseline_test_rank_corr=0.1494，combined_test_rank_corr=0.1471，increment=-0.0023。
- risk: combined_test_abs_return_rank_corr=0.0793。
- nonoverlap: finite_offset_count=5，positive_offset_ratio=0.8000，rank_corr_range=[-0.1739, 0.4843]。
- best_feature: residual_dispersion_5，test_rank_corr=0.1841；这是 test 上的后验诊断，不用于候选选择或晋升。
- qlib_analyzer: qlib.contrib.evaluate.risk_analysis，sidecar=qlib_risk_analysis.csv。
- promotion_decision: not_promoted_missing_pit_holdings_and_no_incremental_evidence。
- 反思: 成分股高振动而 VIX 平静首先是风险状态，不自动给出方向。当前预定义交互相对 weighted_residual 不具样本外增量。
- next_action: 先重新定义或约束交互并取得 validation/test 增量，同时补齐 point-in-time SOXX holdings；两项都通过后才重跑 validated card。
- risk_notes: 当前成分回填存在 survivorship bias；VIX 是 SPX 隐含波动率，不是半导体板块隐含波动率。
- recorder: bb84822de8154b539a8831a86bb296aa。
