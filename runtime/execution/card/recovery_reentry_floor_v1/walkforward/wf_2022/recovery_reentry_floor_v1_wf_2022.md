# recovery_reentry_floor_v1_wf_2022

Walk-forward fixed-actor recovery/reentry floor counterfactual for wf_2022.

- Slow path: `balanced_rule_residual_v19_reg_wf_2022`
- Fast usage path: `balanced_rule_residual_v22_wf_2022`
- Composed path: `recovery_reentry_floor_v1_wf_2022`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_wf_2022 | -47.26% | -1.357 | 49.52% | 0.273 | 182 |
| balanced_rule_residual_v22_wf_2022 | -44.11% | -1.500 | 45.95% | 0.228 | 181 |
| recovery_reentry_floor_v1_wf_2022 | -22.79% | -0.650 | 26.10% | 0.212 | 20 |

## Slices

| slice | balanced_rule_residual_v19_reg_wf_2022 | balanced_rule_residual_v22_wf_2022 | recovery_reentry_floor_v1_wf_2022 |
| --- | --- | --- | --- |
| test_2022 | -47.26% / -1.357 / 49.52% / pos 0.273 / trades 182 | -44.11% / -1.500 / 45.95% / pos 0.228 / trades 181 | -22.79% / -0.650 / 26.10% / pos 0.212 / trades 20 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 24.47pp | 0.707 | -23.43pp | -0.062 |
| test_2022 | 24.47pp | 0.707 | -23.43pp | -0.062 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 21.31pp | 0.850 | -19.85pp | -0.016 |
| test_2022 | 21.31pp | 0.850 | -19.85pp | -0.016 |
