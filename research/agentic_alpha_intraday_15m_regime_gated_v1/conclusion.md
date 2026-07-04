# 中文结论

- 做了什么：在上一版 5m/15m 结果基础上收缩到 15min，新增 regime_v1 特征和交易 gate，并用 qlib.init、D.features、DatasetH/DataHandlerLP、R.start、LGBMClassifier、XGBClassifier 重新训练。
- 核心指标：validation 选择 15min / lgb_xgb_consensus，valid return_pct=19.7527、test return_pct=15.1994、test max_drawdown_pct=-5.9226、test 20bp return_pct=4.5594。
- regime/window 差异：selection 只用 valid 与 2024/2025 historical walk-forward 稳定性，2026 test 只作为报告；regime gate 过滤低信息 bar，保留 time-of-day 标准化量能/波动、SOXL-vs-3x-SOXX 偏离和日内趋势状态。
- 是否晋升：不晋升；decision=not_promoted_walk_forward_not_proven。10bp 是主回测成本，20bp 是压力测试，即便收益改善，也还缺少 bid/ask replay、排队成交、延迟和实盘滑点证明。
- 风险：信号仍是假设 bar 收盘可计算、下一根 bar 开盘成交、下一根 bar 收盘退出；没有证明极端半导体去杠杆日能按回测价格成交。
- 下一步：只把 15min regime gate 作为候选执行层，继续做 bid/ask replay、开盘/收盘 bar 分层、以及 SOXX 成分股/NVDA/AVGO/AMD 分钟级外部状态特征。
- qlib recorder：864b217daec246129600f4c968be6789
