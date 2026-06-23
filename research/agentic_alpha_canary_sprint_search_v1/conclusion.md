# 结论

- what ran: 使用 qlib.init + D.features 从 /Users/rziiiii/dev/workspace/data/qlib/soxl 读取 SOXL/SOXX/VIX，测试 canary sprint、dip sprint、guarded sprint budget 三个 R-G-E-V-D 分支。
- key_metrics: production baseline return=6570.02%, maxDD=-85.23%; 本卡 candidate_rows=11271, passes_both_count=0。
- regime/window differences: 短持有与正自相关过滤能制造高收益尾部，但高收益候选仍穿透 -20% MaxDD；drawdown-gated 候选收益远低于 baseline。
- promotion_decision: not_promoted，未找到同时满足收益超过现有生产策略且 MaxDD 小于 20% 的候选。
- risks: 该分支依赖 SOXX canary 与 SOXL 自身短期状态，无法证明外部 rate/credit lead-lag 对 SOXL 有足够强的因果收益。
- next_action: 继续寻找非同源外部 canary 或更强的日内/事件驱动信号；recorder_id=f4bc3298eea142ba9b0b2fb3de6614cf。
