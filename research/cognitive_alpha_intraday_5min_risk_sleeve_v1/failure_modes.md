# Failure Modes

- 本卡不是新的日内主 alpha；它保留日线 next-open 入场，只验证 5m 风控减仓/退出是否能减少买入后下杀。
- 15m LGB/XGB/consensus 的 label 是 triple-barrier 1h/2h execution improvement，不等同于 open-to-open 收益预测。
- 5m 阈值只从 validation 年份生成；若 test 年效果差，不能用 test 重新调阈值。
- 当前仍缺 bid/ask、盘口深度、延迟和真实成交容量，不能据此直接实盘。
- 若 20/50bp 年度 retention、max drawdown 和风险降低不能同时过门槛，应继续作为诊断卡而非晋升。
