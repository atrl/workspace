# Failure Modes

- fixed_bellwether_proxy: 当前固定五龙头回填历史存在生存偏差，不代表点时SOXX持仓。
- external_intraday: 2025-01-02前没有完整五龙头15分钟覆盖，状态为blocked_insufficient_history。
- sparse_outer_years: 年度事件少于5个时不用于方向一致性晋升。
- factor_only: 结果衡量breakdown辨识力，不等同于含成本仓位收益。
