# v19reg_v22_budget_usage_v1

Fixed-path budget-usage diagnostic with fresh actor daily exports: balanced_rule_residual_v19_reg sets the slow risk-mode band and balanced_rule_residual_v22 is interpreted as usage inside that band.

- Slow path: `balanced_rule_residual_v19_reg_fresh`
- Fast usage path: `balanced_rule_residual_v22_fresh`
- Composed path: `v19reg_v22_budget_usage_v1`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_fresh | 1649.05% | 1.149 | 43.53% | 0.450 | 1428 |
| balanced_rule_residual_v22_fresh | 1128.31% | 0.973 | 53.21% | 0.501 | 1436 |
| v19reg_v22_budget_usage_v1 | 2264.14% | 1.260 | 40.89% | 0.437 | 185 |

## Slices

| slice | balanced_rule_residual_v19_reg_fresh | balanced_rule_residual_v22_fresh | v19reg_v22_budget_usage_v1 |
| --- | --- | --- | --- |
| test_2020 | 128.61% / 1.697 / 28.64% / pos 0.500 / trades 231 | 108.04% / 1.363 / 44.57% / pos 0.561 / trades 231 | 131.95% / 1.774 / 25.19% / pos 0.491 / trades 27 |
| test_2022 | -31.32% / -0.606 / 43.01% / pos 0.328 / trades 235 | -41.73% / -0.738 / 52.84% / pos 0.399 / trades 240 | -21.20% / -0.458 / 31.09% / pos 0.238 / trades 25 |
| test_2023_2024 | 189.57% / 1.454 / 29.01% / pos 0.477 / trades 466 | 193.08% / 1.370 / 32.26% / pos 0.530 / trades 458 | 244.80% / 1.606 / 30.67% / pos 0.475 / trades 66 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 615.09pp | 0.111 | -2.65pp | -0.012 |
| test_2020 | 3.34pp | 0.077 | -3.45pp | -0.009 |
| test_2022 | 10.12pp | 0.148 | -11.92pp | -0.090 |
| test_2023_2024 | 55.23pp | 0.152 | 1.65pp | -0.002 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 1135.83pp | 0.287 | -12.33pp | -0.063 |
| test_2020 | 23.91pp | 0.411 | -19.38pp | -0.071 |
| test_2022 | 20.54pp | 0.280 | -21.75pp | -0.161 |
| test_2023_2024 | 51.72pp | 0.236 | -1.59pp | -0.055 |
