# Failure Modes

- 本地 fallback 只用 SOXL/SOXX 15m 特征；没有 QQQ/SMH/成分股分钟级上下文，因此不能证明外部 lead-lag 因子。
- 每个年度 fold 独立训练 LGB/XGB/consensus，valid 只用于选择 intraday 模型和 delayed policy，test 只报告。
- adverse_event_risk_head 单独训练买入后同日 open-buy-dump 风险，只用 train 拟合并只给 valid/test 输出风险分数；仍不是晋升信号。
- 日内模型的目标是 1h/2h triple-barrier execution improvement，不是日线 open-to-open 趋势收益；若它跳过入场，可能降低回撤但也会丢掉日线 alpha。
- same_day_close 是可交易时序更纯的候选，但仍缺 bid/ask、收盘成交容量、延迟、盘前盘后和极端波动 replay。
- 晋升必须同时满足 2023-2026 年度 20/50bp 的 85%-95% retention、max_dd>-18%、且降低 open-buy-dump 或 crash 暴露。
