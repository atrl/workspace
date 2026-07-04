# 中文结论

- 做了什么：新增 5m 日内风控 sleeve。日线 open-to-open 模型仍按 next-open 入场；15m LGB/XGB/consensus triple-barrier 模型提供风险分数，5m qlib bar 只用于入场后同日触发全平或半仓减仓 replay。
- 为什么日内模型没有超过日线模型：日线模型吃的是 open-to-open/隔夜/趋势段收益；日内模型学的是 1h/2h 执行改善。把它做入场 gate 会漏掉大趋势日，所以本卡改为保留入场、只控制买入后下杀。
- 年度 walk-forward：2023-2026 均按 train_to_T-2 / valid_T-1 / test_T；20/50bp 同时过 retention 85%-95% 与 max_dd>-18% 的年份数=0；降低 open-buy-dump 或 crash 暴露的年份数=2。
- 最新 fold：wf_2026 选 risk_reduce_quarter_5m + drawdown_only，20bp test retention=87.16%，test max_dd=-16.83%，open-buy-dump reduction=0。
- 是否晋升：not_promoted_5min_risk_sleeve_target_gate_failed。若仍不过门槛，下一步不该继续调 test 阈值，而是补更长外部分钟上下文、bid/ask replay 和更明确的风险 label。
- qlib recorder：c51242725bc74fba9e0c8ba6bccdd6ed
