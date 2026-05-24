# recovery_reentry_anchor_confirm_v1_wf_2020

Walk-forward recovery/reentry anchor-confirmed floor counterfactual for wf_2020.

- Slow path: `balanced_rule_residual_v19_reg_wf_2020`
- Fast usage path: `balanced_rule_residual_v22_wf_2020`
- Composed path: `recovery_reentry_anchor_confirm_v1_wf_2020`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_wf_2020 | 69.45% | 1.186 | 28.64% | 0.387 | 171 |
| balanced_rule_residual_v22_wf_2020 | 61.42% | 1.077 | 34.35% | 0.402 | 177 |
| recovery_reentry_anchor_confirm_v1_wf_2020 | 113.89% | 1.583 | 26.14% | 0.424 | 29 |

## Slices

| slice | balanced_rule_residual_v19_reg_wf_2020 | balanced_rule_residual_v22_wf_2020 | recovery_reentry_anchor_confirm_v1_wf_2020 |
| --- | --- | --- | --- |
| test_2020 | 69.45% / 1.186 / 28.64% / pos 0.387 / trades 171 | 61.42% / 1.077 / 34.35% / pos 0.402 / trades 177 | 113.89% / 1.583 / 26.14% / pos 0.424 / trades 29 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 44.44pp | 0.397 | -2.50pp | 0.037 |
| test_2020 | 44.44pp | 0.397 | -2.50pp | 0.037 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 52.47pp | 0.506 | -8.21pp | 0.022 |
| test_2020 | 52.47pp | 0.506 | -8.21pp | 0.022 |
