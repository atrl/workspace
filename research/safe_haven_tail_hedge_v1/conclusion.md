# Safe Haven Tail Hedge V1 Conclusion

- 做了什么：用 qlib SOXL 日线 open/close 和 Polygon/Massive 30DTE OTM put 日线聚合数据，测试 put_80/put_70/put_60 的 Safe Haven 式尾部对冲诊断，recorder id=c42c9ae118a64e6d831cd297aa6a0ddb。
- 核心指标：SOXL open-close 样本收益 73.12%，最大回撤 -43.87%；7/2 SOXL open-close -17.22%，put_80 扣 3% 往返成本后 28.08%。
- 凸性结论：预先持有到 7/2 的最高净倍数只有 2.17x，低于 10.00x 的避风港筛选线，不是 Spitznagel/Safe Haven 意义上的极端凸性。
- 回测差异：最佳日度 funded sleeve 为 2.0%_funded_put_80_daily_cost3%，收益 61.13%，最大回撤 -42.54%，相比 SOXL 收益差 -11.98pp。
- 晋升决定：not_promoted_convexity_insufficient。该方案可以作为风险诊断，不适合作为当前 SOXL 主策略对冲模块。
- 风险：日线期权聚合没有 bid/ask、IV、Greeks 和可成交深度；深 OTM 档在 7/2 的成交覆盖不足；后续必须用分钟级 NBBO/quote 或更长历史期权链验证。
- 下一步：不要继续优化普通保护性 put；应寻找更便宜的远 OTM/更长周期/跨标的 crash proxy，目标是小预算在真实尾部事件中达到 10x+ 净倍数，并用 quote 级数据做成交约束。
