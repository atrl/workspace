# recovery_reentry_cppi_floor_v1_wf_2022

Walk-forward replay for notebook-selected recovery_reentry_cppi_floor_v1 on wf_2022.

- Slow path: `balanced_rule_residual_v19_reg_wf_2022`
- Fast usage path: `balanced_rule_residual_v22_wf_2022`
- Composed path: `recovery_reentry_cppi_floor_v1_wf_2022`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_wf_2022 | -47.26% | -1.357 | 49.52% | 0.273 | 182 |
| balanced_rule_residual_v22_wf_2022 | -44.11% | -1.500 | 45.95% | 0.228 | 181 |
| recovery_reentry_cppi_floor_v1_wf_2022 | -13.22% | -0.391 | 16.03% | 0.151 | 16 |

## Slices

| slice | balanced_rule_residual_v19_reg_wf_2022 | balanced_rule_residual_v22_wf_2022 | recovery_reentry_cppi_floor_v1_wf_2022 |
| --- | --- | --- | --- |
| test_2022 | -47.26% / -1.357 / 49.52% / pos 0.273 / trades 182 | -44.11% / -1.500 / 45.95% / pos 0.228 / trades 181 | -13.22% / -0.391 / 16.03% / pos 0.151 / trades 16 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 34.04pp | 0.966 | -33.49pp | -0.122 |
| test_2022 | 34.04pp | 0.966 | -33.49pp | -0.122 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 30.89pp | 1.109 | -29.92pp | -0.077 |
| test_2022 | 30.89pp | 1.109 | -29.92pp | -0.077 |
