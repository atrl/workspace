# 中文结论

- 做了什么：建立本地 SOXL/SOXX 15m fallback overlay。日线 LGB/XGB open-to-open 模型仍决定 target_position；本卡每个年度 fold 重新训练本地 15m LGB/XGB/consensus triple-barrier 模型，并额外训练同日 open-buy-dump adverse-event 风险头，再用 validation-only delayed policy 决定是否延迟入场或 same_day_close。
- 为什么之前日内模型没有超过日线模型：日内 label 学的是 1h/2h 执行改善，不是多日 open-to-open 趋势；延迟入场能规避部分开盘后下杀，但也会错过 SOXL 日线模型主要依赖的隔夜和趋势段收益。
- 年度 walk-forward：2023-2026 都完成 train_to_T-2 / valid_T-1 / test_T；20/50bp 同时过 85%-95% retention 与 max_dd>-18% 的年份数=3；同时降低 open-buy-dump 或 crash 暴露的年份数=4。
- 最新 fold：wf_2026 选 lgb + delayed_signal_after_60m，20bp test retention=64.72%，test max_dd=-27.66%，open-buy-dump reduction=2。
- 是否晋升：not_promoted_local_intraday_target_gate_failed。当前优化方向应该是保留日线模型作为主仓位，把日内层当作风险/执行 sleeve；要超过日线模型，需要更稳定的外部 lead-lag/成分股分钟上下文或 bid/ask replay 证据，而不是继续用短样本阈值调参。
- qlib recorder：0f2aa26d19ac44b889397897ec890060
