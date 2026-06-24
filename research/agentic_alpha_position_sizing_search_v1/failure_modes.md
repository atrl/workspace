# Failure Modes

- 本卡把 ensemble 转成单一 SOXL 目标仓位口径，不能直接复现旧卡的分支收益合成 8775% 指标。
- selected policy 超过 soxl_backtest YTD，但未超过 raw target-position reference；因此不直接晋升生产。
- cap/hysteresis 降低了 >100% 暴露和隔日反向交易，但牺牲一部分收益。
- latest signal overlay 使用 macro-credit stale/forward-filled 分支，只能读当前仓位，不能作为最新收益验证。
- 浅层模型门控减少回撤但收益显著下降，神经网络仓位器需要更多样本和 walk-forward 约束后再考虑。
