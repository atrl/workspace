# recovery_reentry_floor_v1_wf_2020

Walk-forward fixed-actor recovery/reentry floor counterfactual for wf_2020.

- Slow path: `balanced_rule_residual_v19_reg_wf_2020`
- Fast usage path: `balanced_rule_residual_v22_wf_2020`
- Composed path: `recovery_reentry_floor_v1_wf_2020`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_wf_2020 | 69.45% | 1.186 | 28.64% | 0.387 | 171 |
| balanced_rule_residual_v22_wf_2020 | 61.42% | 1.077 | 34.35% | 0.402 | 177 |
| recovery_reentry_floor_v1_wf_2020 | 108.00% | 1.535 | 26.14% | 0.434 | 31 |

## Slices

| slice | balanced_rule_residual_v19_reg_wf_2020 | balanced_rule_residual_v22_wf_2020 | recovery_reentry_floor_v1_wf_2020 |
| --- | --- | --- | --- |
| test_2020 | 69.45% / 1.186 / 28.64% / pos 0.387 / trades 171 | 61.42% / 1.077 / 34.35% / pos 0.402 / trades 177 | 108.00% / 1.535 / 26.14% / pos 0.434 / trades 31 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 38.55pp | 0.349 | -2.50pp | 0.046 |
| test_2020 | 38.55pp | 0.349 | -2.50pp | 0.046 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 46.58pp | 0.458 | -8.21pp | 0.032 |
| test_2020 | 46.58pp | 0.458 | -8.21pp | 0.032 |
