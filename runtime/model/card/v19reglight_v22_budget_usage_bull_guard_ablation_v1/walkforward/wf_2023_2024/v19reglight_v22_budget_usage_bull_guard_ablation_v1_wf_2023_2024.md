# v19reglight_v22_budget_usage_bull_guard_ablation_v1_wf_2023_2024

No-retrain slow-path ablation: balanced_rule_residual_v19_reg_light sets the slow risk mode, balanced_rule_residual_v22 supplies fast usage, and the approved bull participation guard is applied.

- Slow path: `balanced_rule_residual_v19_reg_light_wf_2023_2024`
- Fast usage path: `balanced_rule_residual_v22_wf_2023_2024`
- Composed path: `v19reglight_v22_budget_usage_bull_guard_ablation_v1_wf_2023_2024`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_light_wf_2023_2024 | 212.36% | 1.325 | 39.80% | 0.554 | 396 |
| balanced_rule_residual_v22_wf_2023_2024 | 32.56% | 0.866 | 33.21% | 0.362 | 178 |
| v19reglight_v22_budget_usage_bull_guard_ablation_v1_wf_2023_2024 | 33.34% | 0.873 | 36.91% | 0.359 | 24 |

## Slices

| slice | balanced_rule_residual_v19_reg_light_wf_2023_2024 | balanced_rule_residual_v22_wf_2023_2024 | v19reglight_v22_budget_usage_bull_guard_ablation_v1_wf_2023_2024 |
| --- | --- | --- | --- |
| test_2023_2024 | 212.36% / 1.325 / 39.80% / pos 0.554 / trades 396 | 32.56% / 0.866 / 33.21% / pos 0.362 / trades 178 | 33.34% / 0.873 / 36.91% / pos 0.359 / trades 24 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | -179.03pp | -0.452 | -2.89pp | -0.195 |
| test_2023_2024 | -179.03pp | -0.452 | -2.89pp | -0.195 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 0.78pp | 0.007 | 3.70pp | -0.003 |
| test_2023_2024 | 0.78pp | 0.007 | 3.70pp | -0.003 |
