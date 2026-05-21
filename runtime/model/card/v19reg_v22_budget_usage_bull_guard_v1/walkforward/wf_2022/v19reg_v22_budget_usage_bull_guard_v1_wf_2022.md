# v19reg_v22_budget_usage_bull_guard_v1_wf_2022

Bull-participation budget-usage diagnostic: v19_reg sets the slow risk mode, v22 supplies usage, and a shallow strong-trend guard floors the raw target to the fast v22 position.

- Slow path: `balanced_rule_residual_v19_reg_wf_2022`
- Fast usage path: `balanced_rule_residual_v22_wf_2022`
- Composed path: `v19reg_v22_budget_usage_bull_guard_v1_wf_2022`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_wf_2022 | -47.26% | -1.357 | 49.52% | 0.273 | 182 |
| balanced_rule_residual_v22_wf_2022 | -44.11% | -1.500 | 45.95% | 0.228 | 181 |
| v19reg_v22_budget_usage_bull_guard_v1_wf_2022 | -20.68% | -0.624 | 26.65% | 0.184 | 15 |

## Slices

| slice | balanced_rule_residual_v19_reg_wf_2022 | balanced_rule_residual_v22_wf_2022 | v19reg_v22_budget_usage_bull_guard_v1_wf_2022 |
| --- | --- | --- | --- |
| test_2022 | -47.26% / -1.357 / 49.52% / pos 0.273 / trades 182 | -44.11% / -1.500 / 45.95% / pos 0.228 / trades 181 | -20.68% / -0.624 / 26.65% / pos 0.184 / trades 15 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 26.58pp | 0.733 | -22.87pp | -0.090 |
| test_2022 | 26.58pp | 0.733 | -22.87pp | -0.090 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 23.42pp | 0.875 | -19.30pp | -0.044 |
| test_2022 | 23.42pp | 0.875 | -19.30pp | -0.044 |
