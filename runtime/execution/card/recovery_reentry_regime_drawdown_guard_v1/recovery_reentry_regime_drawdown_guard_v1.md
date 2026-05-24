# recovery_reentry_regime_drawdown_guard_v1

Notebook-selected regime-aware account-equity drawdown controller. It keeps the anchor-confirmed reentry guard, but applies the equity drawdown cap only when the account is in drawdown and long-anchor/high-volatility downside filters are active.

- Slow path: `balanced_rule_residual_v19_reg_current`
- Fast usage path: `balanced_rule_residual_v22_current`
- Composed path: `recovery_reentry_regime_drawdown_guard_v1`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_current | 1667.81% | 1.151 | 43.53% | 0.450 | 1430 |
| balanced_rule_residual_v22_current | 1141.54% | 0.975 | 53.21% | 0.501 | 1437 |
| recovery_reentry_regime_drawdown_guard_v1 | 2272.85% | 1.248 | 32.12% | 0.460 | 212 |

## Slices

| slice | balanced_rule_residual_v19_reg_current | balanced_rule_residual_v22_current | recovery_reentry_regime_drawdown_guard_v1 |
| --- | --- | --- | --- |
| test_2020 | 128.61% / 1.697 / 28.64% / pos 0.500 / trades 231 | 108.04% / 1.363 / 44.57% / pos 0.561 / trades 231 | 154.60% / 1.840 / 25.27% / pos 0.536 / trades 30 |
| test_2022 | -31.32% / -0.606 / 43.01% / pos 0.328 / trades 235 | -41.73% / -0.738 / 52.84% / pos 0.399 / trades 240 | -10.55% / -0.115 / 21.99% / pos 0.240 / trades 30 |
| test_2023_2024 | 189.57% / 1.454 / 29.01% / pos 0.477 / trades 466 | 193.08% / 1.370 / 32.26% / pos 0.530 / trades 458 | 209.14% / 1.453 / 30.67% / pos 0.497 / trades 70 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 605.05pp | 0.097 | -11.41pp | 0.010 |
| test_2020 | 25.99pp | 0.143 | -3.37pp | 0.036 |
| test_2022 | 20.77pp | 0.492 | -21.02pp | -0.087 |
| test_2023_2024 | 19.57pp | -0.001 | 1.65pp | 0.020 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 1131.31pp | 0.273 | -21.09pp | -0.041 |
| test_2020 | 46.55pp | 0.477 | -19.30pp | -0.025 |
| test_2022 | 31.19pp | 0.624 | -30.85pp | -0.159 |
| test_2023_2024 | 16.06pp | 0.083 | -1.59pp | -0.033 |
