# v19reglight_v22_budget_usage_bull_guard_ablation_v1_wf_2022

No-retrain slow-path ablation: balanced_rule_residual_v19_reg_light sets the slow risk mode, balanced_rule_residual_v22 supplies fast usage, and the approved bull participation guard is applied.

- Slow path: `balanced_rule_residual_v19_reg_light_wf_2022`
- Fast usage path: `balanced_rule_residual_v22_wf_2022`
- Composed path: `v19reglight_v22_budget_usage_bull_guard_ablation_v1_wf_2022`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_light_wf_2022 | -37.25% | -1.336 | 39.18% | 0.206 | 179 |
| balanced_rule_residual_v22_wf_2022 | -44.11% | -1.500 | 45.95% | 0.228 | 181 |
| v19reglight_v22_budget_usage_bull_guard_ablation_v1_wf_2022 | -23.22% | -1.231 | 24.77% | 0.114 | 22 |

## Slices

| slice | balanced_rule_residual_v19_reg_light_wf_2022 | balanced_rule_residual_v22_wf_2022 | v19reglight_v22_budget_usage_bull_guard_ablation_v1_wf_2022 |
| --- | --- | --- | --- |
| test_2022 | -37.25% / -1.336 / 39.18% / pos 0.206 / trades 179 | -44.11% / -1.500 / 45.95% / pos 0.228 / trades 181 | -23.22% / -1.231 / 24.77% / pos 0.114 / trades 22 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 14.03pp | 0.105 | -14.41pp | -0.092 |
| test_2022 | 14.03pp | 0.105 | -14.41pp | -0.092 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 20.89pp | 0.269 | -21.18pp | -0.113 |
| test_2022 | 20.89pp | 0.269 | -21.18pp | -0.113 |
