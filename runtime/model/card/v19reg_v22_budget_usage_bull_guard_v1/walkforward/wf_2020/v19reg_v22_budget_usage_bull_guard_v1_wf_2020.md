# v19reg_v22_budget_usage_bull_guard_v1_wf_2020

Bull-participation budget-usage diagnostic: v19_reg sets the slow risk mode, v22 supplies usage, and a shallow strong-trend guard floors the raw target to the fast v22 position.

- Slow path: `balanced_rule_residual_v19_reg_wf_2020`
- Fast usage path: `balanced_rule_residual_v22_wf_2020`
- Composed path: `v19reg_v22_budget_usage_bull_guard_v1_wf_2020`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_wf_2020 | 69.45% | 1.186 | 28.64% | 0.387 | 171 |
| balanced_rule_residual_v22_wf_2020 | 61.42% | 1.077 | 34.35% | 0.402 | 177 |
| v19reg_v22_budget_usage_bull_guard_v1_wf_2020 | 98.27% | 1.456 | 26.14% | 0.399 | 25 |

## Slices

| slice | balanced_rule_residual_v19_reg_wf_2020 | balanced_rule_residual_v22_wf_2020 | v19reg_v22_budget_usage_bull_guard_v1_wf_2020 |
| --- | --- | --- | --- |
| test_2020 | 69.45% / 1.186 / 28.64% / pos 0.387 / trades 171 | 61.42% / 1.077 / 34.35% / pos 0.402 / trades 177 | 98.27% / 1.456 / 26.14% / pos 0.399 / trades 25 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 28.82pp | 0.270 | -2.50pp | 0.012 |
| test_2020 | 28.82pp | 0.270 | -2.50pp | 0.012 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 36.85pp | 0.379 | -8.21pp | -0.003 |
| test_2020 | 36.85pp | 0.379 | -8.21pp | -0.003 |
