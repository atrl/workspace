# SIGNED_TE_REVERSION

**方向**: factor
**类别**: transition / mean-reversion
**状态**: 🟡 Candidate / blocked

## 假设

SOXL 相对 `3 * SOXX` 的有符号 tracking error 类似币圈 futures basis / funding dislocation。
当 SOXL 显著跑输 3x underlying path 时，`SIGNED_TE_REVERSION_*` 转正，预期后续存在均值回归反弹。

## 方法/配置

公式: `-zscore(rolling_sum(SOXL_return - 3 * SOXX_return, window), 252d)`

测试窗口: 5d / 10d / 21d

## 结果

| 指标 | 数值 |
|------|------|
| Best IC Feature | SIGNED_TE_REVERSION_21 |
| Best IC Horizon | 20d |
| IC Mean | +0.014186 |
| IC Positive Ratio | 0.516 |
| IC t-stat | 2.459 |
| Best Backtest Feature | SIGNED_TE_REVERSION_10 |
| Threshold | 0.50 |
| Sharpe | +0.561 |
| Profit Factor | 1.228 |
| Signal Accuracy | 0.546 |
| Best Feature Core Correlation | 0.577 |
| Max Core Correlation | 0.811 |

## 结论

该因子 standalone backtest 有正收益，但 IC mean / positive ratio 未过门槛，方向稳定性不足。
`SIGNED_TE_REVERSION_5` 还超过 core correlation budget；本轮不加入 core RL feature schema。
后续若继续，应只围绕 `SIGNED_TE_REVERSION_10` 做 regime-gated ablation，而不是直接推广整组信号。

## 参考

- Crypto futures basis / funding-rate dislocation analogue
- LETF tracking-error mean reversion: compare with `TRANS_TE_ZSCORE`
