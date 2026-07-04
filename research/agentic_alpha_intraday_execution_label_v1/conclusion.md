# 中文结论

- 做了什么：把 15m 日内模型的 label 从下一根 bar 涨跌升级为 triple-barrier + execution-improvement，并加入 open gap、VWAP 距离、realized vol 扩张、前 30/60 分钟趋势质量、量价吸收/派发、SOXX lead-lag 等同日上下文因子。
- 核心指标：validation 选择 lgb，valid return_pct=-21.2448、test return_pct=12.9098、test max_drawdown_pct=-11.2049。
- walk-forward：按 train 到 T-2、valid T-1、test T 记录 2023-2026，正收益比例=0.50。
- 是否晋升：不晋升；decision=not_promoted_walk_forward_not_proven。它是执行层候选，不替代日级 open-to-open 仓位模型。
- 风险：当前分钟级外部上下文只有 SOXL/SOXX，QQQ/SMH 与 NVDA/AVGO/AMD/TSM/ASML 还没有分钟级 qlib store；真实执行还缺 bid/ask replay、排队成交、延迟和极端波动成交验证。
- qlib recorder：13e134d6818f4dc39622b1c3458b979e
