# 结论与反思

- what_ran: 使用 qlib SOXX 成分股和 SOXL/SOXX open 数据，分离单股归因、SOXX 聚合、3x 日复位映射与执行截断。
- data: as_of=2026-07-02，constituents=30，PIT=blocked_current_holdings_backfill。
- factor: stable_aggregate_count=1，single_name_same_sign_ratio=0.0667，best_test_signal=weighted_residual，test_rank_corr=0.1494，test_spread=1.8656%。
- uncertainty: nonoverlap_positive_offset_ratio=0.8000，offset_rank_corr_range=[-0.1789, 0.3409]，observations_per_offset=23-24。
- leverage: test_tracking_rmse=1.2477%，rank_corr=0.9974。
- execution: raw_zero_ratio=0.9839，floor_ratio=1.0000，probability_rank_corr=-0.1885。
- promotion_decision: not_promoted_missing_pit_holdings。
- 反思: 过去把单股代理、SOXX 传导、SOXL 杠杆路径和仓位策略压进同一模型，导致因子失效与执行截断无法区分。本实验只允许形成研究诊断；在补齐 PIT holdings、purged walk-forward 和稳定 SOXX 样本外增量前，不应放宽交易约束。
- next_action: 先补 point-in-time SOXX holdings，再对每只成分股做横截面预测和 leave-one-name-out 归因；通过后才把预测映射为 SOXX core + SOXL overlay。
- risk: 当前权重存在 survivorship bias，五日标签重叠，结果不构成买卖建议。
- recorder: 50ec0f54805242b4b79df4720a25911b。
