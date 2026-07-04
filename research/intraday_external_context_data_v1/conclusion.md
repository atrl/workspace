# 中文结论

- 做了什么：建立外部 15m 上下文数据卡，审计 QQQ/SMH/SOXX 与 NVDA/AVGO/AMD/TSM/ASML 是否可进入 qlib provider。
- 数据状态：ok_partial_insufficient_for_full_walk_forward；可用=['QQQ', 'SMH']；缺失=['AMD', 'ASML', 'AVGO', 'NVDA', 'SOXX', 'TSM']。
- 覆盖范围：请求 2019-01-02 到 2026-07-02，完整 walk-forward 需要 2019-01-02 到 2026-07-02。
- 用途：这张卡只解决外部 15m 上下文数据和缺口审计，不直接证明日内模型可晋升。
- 风险：Polygon 权限、历史窗口和限流会影响可重复性；缺失 symbol 不会被空值因子伪装成 breadth。
- qlib recorder：bd4e85674233402fb59d1c77250c9091