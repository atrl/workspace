# Failure Modes

- Delayed observable overlay 只能在前 30/60 分钟 bar 完成后使用这些特征，不能影响 next-open 或前 4 根 bar 的成交。
- 选择规则只用 validation，并要求 20bp/50bp validation 同时满足 85%-95% retention、距离 retention 边界至少 0.05pct、max_dd>-18%、skip<=15%；eligible exit policy 是 hold_to_next_open 和 same_day_close。
- same_day_close 是纯日内平仓候选，用 entry day 最后一根可用 15m close 退出；它仍缺少 bid/ask、收盘成交容量和延迟 replay。
- retention 目标是 validation 85%-95% hard gate；超过 95% 的短窗口收益只能作为诊断，不算晋级通过。
- 年度验证使用 daily walk-forward replay positions 的 validation/test 片段；train 阶段不要求 selected positions，因为仓位模型训练证据属于源日线 card。
- 当前外部 15m 上下文覆盖仍不足，walk_forward_feasibility.csv 对 2023-2026 年度 fold 继续 fail closed。
- 还没有 bid/ask、队列、延迟和盘前盘后 replay；只能作为研究候选，不是生产执行策略。
