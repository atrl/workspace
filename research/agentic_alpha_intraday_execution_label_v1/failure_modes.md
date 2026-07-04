# Failure Modes

- Label 已从下一根 15m 涨跌升级到 triple-barrier + execution-improvement，但仍是 bar 数据回放，不是 bid/ask 或队列成交。
- 当前 15min qlib provider 只有 SOXL/SOXX；QQQ/SMH 和 NVDA/AVGO/AMD/TSM/ASML 分钟级上下文尚未进入 qlib store。
- Walk-forward 只允许 valid 选阈值，test 只报告；如果某一年失败，不能用 2026 test 表现反向调参。
- 该模型仍然只适合作为日级仓位模型的执行层候选，不应替代日级 open-to-open 趋势仓位模型。
