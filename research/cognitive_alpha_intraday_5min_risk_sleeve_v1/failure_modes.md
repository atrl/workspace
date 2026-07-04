# Failure Modes

- 本卡不是新的日内主 alpha；它保留日线 next-open 入场，只验证 5m 风控减仓/退出是否能减少买入后下杀。
- 15m LGB/XGB/consensus 的 label 是 triple-barrier 1h/2h execution improvement，不等同于 open-to-open 收益预测。
- 5m risk_score 字段沿用 15m 执行改善概率；风控触发方向是低概率触发，而不是高风险概率触发。
- reenter-close 变体只降低触发日至收盘的日内风险，收盘恢复目标仓位以保留日线模型的隔夜段。
- 5m 阈值只从 validation 年份生成；若 test 年效果差，不能用 test 重新调阈值。
- 5m context stress 候选包含 SOXL/SOXX 本地 qlib bar 的 VWAP、SOXX 确认、波动扩张、杠杆偏离、gap reclaim、趋势质量和量价吸收/派发。
- 外部 QQQ/SMH/SOXX lead-lag 与 NVDA/AVGO/AMD/TSM/ASML breadth 已按 15m bar 向后对齐到 5m replay，但外部历史不足完整 annual walk-forward 前只能作为诊断候选。
- 风控 replay 已避免同一根 5m bar 收盘触发又按同一收盘成交；当前仍缺 bid/ask、盘口深度和真实成交容量，不能据此直接实盘。
- 若 20/50bp 年度 retention、max drawdown 和风险降低不能同时过门槛，应继续作为诊断卡而非晋升。
