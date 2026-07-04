# 中文结论

- 做了什么：建立外部 15m 上下文数据卡，审计 QQQ/SMH/SOXX 与 NVDA/AVGO/AMD/TSM/ASML 是否可进入 qlib provider。
- 数据状态：ok_partial_insufficient_for_full_walk_forward；可用=['AMD', 'ASML', 'AVGO', 'NVDA', 'QQQ', 'SMH', 'SOXX', 'TSM']；缺失=[]。
- 覆盖范围：请求 2019-01-02 到 2026-07-02，完整 walk-forward 需要 2019-01-02 到 2026-07-02。
- 覆盖缺口：任一 symbol 最早 regular bar=2024-07-05 09:45:00；全 symbol 完整 regular 起点=2025-01-02 09:45:00；缺失年份=[2019, 2020, 2021, 2022, 2023, 2024]。
- provider 阻塞：{'blocked_no_rows': 1, 'blocked_provider_history': 4, 'blocked_provider_rate_limit': 8}；下一步=upgrade_polygon_history_plan_or_load_alternate_15min_source。
- 用途：这张卡只解决外部 15m 上下文数据和缺口审计，不直接证明日内模型可晋升。
- 风险：Polygon 权限、历史窗口和限流会影响可重复性；缺失 symbol 不会被空值因子伪装成 breadth。
- qlib recorder：97f11fc996df4fb898ce7b613a9bff4f