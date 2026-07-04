# 中文结论

- 做了什么：新增 5m 日内风控 sleeve。日线 open-to-open 模型仍按 next-open 入场；15m LGB/XGB/consensus triple-barrier 模型提供风险分数，5m qlib bar 在收盘触发后按下一根 5m 开盘做全平或半仓减仓 replay；本轮新增 SOXL/SOXX 5m VWAP、SOXX 确认、波动扩张、杠杆偏离、gap reclaim、趋势质量和量价吸收/派发 context stress 候选，并接入 QQQ/SMH/SOXX 15m lead-lag 与半导体成分股 breadth 诊断候选。
- 外部 context 覆盖：5m replay 可对齐行数=28392，full_walk_forward_history_available=False；历史不足完整 2023-2026 annual walk-forward 前，外部因子只能解释/诊断，不能作为晋升门槛放宽依据。
- 为什么日内模型没有超过日线模型：日线模型吃的是 open-to-open/隔夜/趋势段收益；日内模型学的是 1h/2h 执行改善。把它做入场 gate 会漏掉大趋势日，所以本卡改为保留入场、只控制买入后下杀。
- 年度 walk-forward：2023-2026 均按 train_to_T-2 / valid_T-1 / test_T；20/50bp 同时过 retention 85%-95% 与 max_dd>-18% 的年份数=0；降低 open-buy-dump 或 crash 暴露的年份数=2。
- 最新 fold：wf_2026 选 risk_reduce_quarter_5m + context_stress_or_drawdown，20bp test retention=80.48%，test max_dd=-16.67%，open-buy-dump reduction=0。
- 是否晋升：not_promoted_5min_risk_sleeve_target_gate_failed。若仍不过门槛，下一步不该继续调 test 阈值，而是补更长外部分钟上下文、bid/ask replay 和更明确的风险 label。
- qlib recorder：7a7ddfb2cb5c408e9e768a56f66d7549
