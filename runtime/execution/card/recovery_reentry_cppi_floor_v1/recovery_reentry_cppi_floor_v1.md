# recovery_reentry_cppi_floor_v1

Notebook-selected CPPI/TIPP-style lower drawdown floor. It keeps anchor-confirmed recovery reentry and the same broken-anchor/high-volatility drawdown regime filters, but screens lower min_target values to test whether the remaining strict-OOS drawdown can move toward 15% without walk-forward overdefense.

- Slow path: `balanced_rule_residual_v19_reg_current`
- Fast usage path: `balanced_rule_residual_v22_current`
- Composed path: `recovery_reentry_cppi_floor_v1`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_current | 1667.81% | 1.151 | 43.53% | 0.450 | 1430 |
| balanced_rule_residual_v22_current | 1141.54% | 0.975 | 53.21% | 0.501 | 1437 |
| recovery_reentry_cppi_floor_v1 | 2410.99% | 1.286 | 32.12% | 0.440 | 208 |

## Slices

| slice | balanced_rule_residual_v19_reg_current | balanced_rule_residual_v22_current | recovery_reentry_cppi_floor_v1 |
| --- | --- | --- | --- |
| test_2020 | 128.61% / 1.697 / 28.64% / pos 0.500 / trades 231 | 108.04% / 1.363 / 44.57% / pos 0.561 / trades 231 | 154.60% / 1.840 / 25.27% / pos 0.536 / trades 30 |
| test_2022 | -31.32% / -0.606 / 43.01% / pos 0.328 / trades 235 | -41.73% / -0.738 / 52.84% / pos 0.399 / trades 240 | -9.01% / -0.144 / 19.64% / pos 0.160 / trades 30 |
| test_2023_2024 | 189.57% / 1.454 / 29.01% / pos 0.477 / trades 466 | 193.08% / 1.370 / 32.26% / pos 0.530 / trades 458 | 209.14% / 1.453 / 30.67% / pos 0.497 / trades 70 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 743.18pp | 0.135 | -11.41pp | -0.010 |
| test_2020 | 25.99pp | 0.143 | -3.37pp | 0.036 |
| test_2022 | 22.31pp | 0.463 | -23.37pp | -0.167 |
| test_2023_2024 | 19.57pp | -0.001 | 1.65pp | 0.020 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 1269.45pp | 0.311 | -21.09pp | -0.061 |
| test_2020 | 46.55pp | 0.477 | -19.30pp | -0.025 |
| test_2022 | 32.72pp | 0.595 | -33.20pp | -0.239 |
| test_2023_2024 | 16.06pp | 0.083 | -1.59pp | -0.033 |
