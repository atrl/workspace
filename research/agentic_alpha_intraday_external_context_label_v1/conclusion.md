# 中文结论

- 做了什么：把 15m 日内模型的 label 从下一根 bar 涨跌升级为 triple-barrier + execution-improvement，并加入 open gap、VWAP 距离、realized vol 扩张、前 30/60 分钟趋势质量、量价吸收/派发、SOXX lead-lag 等同日上下文因子。并接入外部 15m 上下文：QQQ/SMH/SOXX lead-lag，NVDA/AVGO/AMD/TSM/ASML 半导体成分股 breadth/equal-weight。
- 核心指标：validation 选择 lgb，valid return_pct=-9.2119、test return_pct=7.1085、test max_drawdown_pct=-16.2333。
- walk-forward：按 train 到 T-2、valid T-1、test T 记录 2023-2026，正收益比例=0.00。
- 是否晋升：不晋升；decision=not_promoted_external_history_insufficient_for_required_walk_forward。它是执行层候选，不替代日级 open-to-open 仓位模型。
- 风险：外部 15m 上下文只有 2025-01-02 之后的数据，完整 train-to-T-2 / valid-T-1 / test-T 的 2019 起步 walk-forward 不满足；真实执行还缺 bid/ask replay、排队成交、延迟和极端波动成交验证。
- qlib recorder：67579fc016ad420a86806b8d8e6f806d
