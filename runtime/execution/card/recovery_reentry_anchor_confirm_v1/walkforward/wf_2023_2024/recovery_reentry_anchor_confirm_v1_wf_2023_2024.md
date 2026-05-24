# recovery_reentry_anchor_confirm_v1_wf_2023_2024

Walk-forward recovery/reentry anchor-confirmed floor counterfactual for wf_2023_2024.

- Slow path: `balanced_rule_residual_v19_reg_wf_2023_2024`
- Fast usage path: `balanced_rule_residual_v22_wf_2023_2024`
- Composed path: `recovery_reentry_anchor_confirm_v1_wf_2023_2024`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_wf_2023_2024 | 42.15% | 1.084 | 30.70% | 0.335 | 180 |
| balanced_rule_residual_v22_wf_2023_2024 | 32.56% | 0.866 | 33.21% | 0.362 | 178 |
| recovery_reentry_anchor_confirm_v1_wf_2023_2024 | 58.19% | 1.326 | 30.18% | 0.337 | 29 |

## Slices

| slice | balanced_rule_residual_v19_reg_wf_2023_2024 | balanced_rule_residual_v22_wf_2023_2024 | recovery_reentry_anchor_confirm_v1_wf_2023_2024 |
| --- | --- | --- | --- |
| test_2023_2024 | 42.15% / 1.084 / 30.70% / pos 0.335 / trades 180 | 32.56% / 0.866 / 33.21% / pos 0.362 / trades 178 | 58.19% / 1.326 / 30.18% / pos 0.337 / trades 29 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 16.04pp | 0.242 | -0.52pp | 0.001 |
| test_2023_2024 | 16.04pp | 0.242 | -0.52pp | 0.001 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 25.64pp | 0.460 | -3.03pp | -0.025 |
| test_2023_2024 | 25.64pp | 0.460 | -3.03pp | -0.025 |
