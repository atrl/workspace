# Failure Modes

- Label 已从下一根 15m 涨跌升级到 triple-barrier + execution-improvement，但仍是 bar 数据回放，不是 bid/ask 或队列成交。
- 外部上下文已接入 QQQ/SMH/SOXX 与 NVDA/AVGO/AMD/TSM/ASML 的 15min qlib store，但历史从 2025 才开始，不能满足 2019 起步的完整 walk-forward。
- Walk-forward 只允许 valid 选阈值，test 只报告；如果某一年失败，不能用 2026 test 表现反向调参。
- 该模型仍然只适合作为日级仓位模型的执行层候选，不应替代日级 open-to-open 趋势仓位模型。
