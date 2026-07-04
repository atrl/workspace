# Failure Modes

- Delayed observable overlay 只能在前 30/60 分钟 bar 完成后使用这些特征，不能影响 next-open 或前 4 根 bar 的成交。
- 选择规则只用 validation，并要求 20bp/50bp validation 同时满足 85%-95% retention、max_dd>-18%、skip<=15%。
- 年度验证使用 daily walk-forward replay positions 的 validation/test 片段；train 阶段不要求 selected positions，因为仓位模型训练证据属于源日线 card。
- 当前外部 15m 上下文覆盖仍不足，walk_forward_feasibility.csv 对 2023-2026 年度 fold 继续 fail closed。
- 还没有 bid/ask、队列、延迟和盘前盘后 replay；只能作为研究候选，不是生产执行策略。
