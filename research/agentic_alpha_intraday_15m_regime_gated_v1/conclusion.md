# 中文结论

- 做了什么：在上一版 5m/15m 结果基础上收缩到 15min，保留 regime_v1 特征，把交易 gate 优化为 regime_v2_dislocation，并明确为 15min 日内执行，并用 qlib.init、D.features、DatasetH/DataHandlerLP、R.start、LGBMClassifier、XGBClassifier 重新训练。
- 核心指标：validation 选择 15min / lgb_xgb_consensus，valid return_pct=48.5795、test return_pct=13.7533、test max_drawdown_pct=-8.6421、test 20bp return_pct=9.0856。
- regime/window 差异：selection 只用 valid 与 2024/2025 historical walk-forward 稳定性，2026 test 只作为报告；regime_v2_dislocation 只交易 SOXL-vs-3x-SOXX 同一根 15m bar 复制偏离超过 0.0012 的状态，过滤低信息普通 bar。
- 是否晋升：不晋升；decision=not_promoted_intraday_replay_missing。10bp 是主回测成本，20bp 是压力测试，即便收益改善，也还缺少 bid/ask replay、排队成交、延迟和实盘滑点证明。
- 风险：信号假设 bar_t 收盘可计算、bar_t+1 开盘进、bar_t+1 收盘出；这是日内执行，不是 open-to-open，但仍没有证明极端半导体去杠杆日能按回测价格成交。
- 下一步：只把 15min regime_v2_dislocation 作为候选日内执行层，继续做 bid/ask replay、开盘/收盘 bar 分层、以及 SOXX 成分股/NVDA/AVGO/AMD 分钟级外部状态特征。
- qlib recorder：19df951d1fa544e3ba64217a6c3d866e
