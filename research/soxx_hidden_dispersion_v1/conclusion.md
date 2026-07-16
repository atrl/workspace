# 结论与反思

- what_ran: 使用 qlib SOXX 成分股 OHLCV、SOXX/SOXL open 与 VIX，构建内部离散度、抵消、相关性破裂、领导轮换和成交量 breadth 特征。
- data: as_of=2026-07-02，constituents=30，PIT=blocked_current_holdings_backfill。
- direction: baseline_test_rank_corr=0.1494，combined_test_rank_corr=0.1473，increment=-0.0021。
- risk: combined_test_abs_return_rank_corr=0.0794。
- nonoverlap: positive_offset_ratio=0.8000，rank_corr_range=[-0.1739, 0.4843]。
- best_feature: residual_dispersion_5，test_rank_corr=0.1841。
- promotion_decision: not_promoted_missing_pit_holdings。
- 反思: 成分股高振动而 VIX 平静首先是风险状态，不自动给出方向。只有与残差动量交互后仍有样本外增量，才适合作为方向 meta-label。
- next_action: 补齐 point-in-time SOXX holdings 后重跑 validated card；在此之前仅保留诊断，不放宽仓位约束。
- risk_notes: 当前成分回填存在 survivorship bias；VIX 是 SPX 隐含波动率，不是半导体板块隐含波动率。
- recorder: eb197b16966a41058286d92b40f50da5。
