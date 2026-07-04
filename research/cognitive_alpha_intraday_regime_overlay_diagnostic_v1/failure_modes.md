# Failure Modes

- Regime overlay 只允许使用 daily close 时已知的 probability、target_position、turnover、raw_target 做条件选择。
- 第一小时走势或 15m 概率不能用于选择前 4 根 15m TWAP；这类条件会造成执行时序泄漏，已从正式选择中排除。
- 即使 validation 有候选过 85%-95% retention 与 max_dd>-18% 门槛，test 仍必须独立报告，不能回填调参。
- 当前外部 15m 历史仍不足以证明完整 2019-2026 walk-forward 推广。
