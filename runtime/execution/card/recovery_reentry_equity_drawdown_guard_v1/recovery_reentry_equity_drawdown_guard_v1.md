# recovery_reentry_equity_drawdown_guard_v1

Fixed-actor anchor-confirmed recovery reentry with account-equity drawdown governor.

- Slow path: `balanced_rule_residual_v19_reg_current`
- Fast usage path: `balanced_rule_residual_v22_current`
- Composed path: `recovery_reentry_equity_drawdown_guard_v1`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_current | 1667.81% | 1.151 | 43.53% | 0.450 | 1430 |
| balanced_rule_residual_v22_current | 1141.54% | 0.975 | 53.21% | 0.501 | 1437 |
| recovery_reentry_equity_drawdown_guard_v1 | 909.85% | 1.075 | 34.37% | 0.390 | 334 |

## Slices

| slice | balanced_rule_residual_v19_reg_current | balanced_rule_residual_v22_current | recovery_reentry_equity_drawdown_guard_v1 |
| --- | --- | --- | --- |
| test_2020 | 128.61% / 1.697 / 28.64% / pos 0.500 / trades 231 | 108.04% / 1.363 / 44.57% / pos 0.561 / trades 231 | 107.58% / 1.663 / 24.37% / pos 0.459 / trades 48 |
| test_2022 | -31.32% / -0.606 / 43.01% / pos 0.328 / trades 235 | -41.73% / -0.738 / 52.84% / pos 0.399 / trades 240 | -22.39% / -0.829 / 28.06% / pos 0.185 / trades 47 |
| test_2023_2024 | 189.57% / 1.454 / 29.01% / pos 0.477 / trades 466 | 193.08% / 1.370 / 32.26% / pos 0.530 / trades 458 | 130.53% / 1.265 / 25.73% / pos 0.409 / trades 128 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | -757.96pp | -0.076 | -9.17pp | -0.059 |
| test_2020 | -21.03pp | -0.033 | -4.27pp | -0.041 |
| test_2022 | 8.93pp | -0.222 | -14.94pp | -0.142 |
| test_2023_2024 | -59.04pp | -0.189 | -3.28pp | -0.068 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | -231.69pp | 0.100 | -18.85pp | -0.110 |
| test_2020 | -0.46pp | 0.301 | -20.20pp | -0.102 |
| test_2022 | 19.35pp | -0.090 | -24.77pp | -0.214 |
| test_2023_2024 | -62.55pp | -0.106 | -6.53pp | -0.121 |
