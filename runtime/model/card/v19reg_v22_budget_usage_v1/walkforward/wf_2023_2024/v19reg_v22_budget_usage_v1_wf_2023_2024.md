# v19reg_v22_budget_usage_v1_wf_2023_2024

Walk-forward budget-usage composition: v19_reg wf_2023_2024 sets the slow risk-mode band and v22 wf_2023_2024 supplies usage inside that band.

- Slow path: `v19_reg_wf_2023_2024`
- Fast usage path: `v22_wf_2023_2024`
- Composed path: `v19reg_v22_budget_usage_v1_wf_2023_2024`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| v19_reg_wf_2023_2024 | 42.15% | 1.084 | 30.70% | 0.335 | 180 |
| v22_wf_2023_2024 | 32.56% | 0.866 | 33.21% | 0.362 | 178 |
| v19reg_v22_budget_usage_v1_wf_2023_2024 | 62.64% | 1.424 | 30.18% | 0.316 | 27 |

## Slices

| slice | v19_reg_wf_2023_2024 | v22_wf_2023_2024 | v19reg_v22_budget_usage_v1_wf_2023_2024 |
| --- | --- | --- | --- |
| test_2023_2024 | 42.15% / 1.084 / 30.70% / pos 0.335 / trades 180 | 32.56% / 0.866 / 33.21% / pos 0.362 / trades 178 | 62.64% / 1.424 / 30.18% / pos 0.316 / trades 27 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 20.49pp | 0.339 | -0.52pp | -0.019 |
| test_2023_2024 | 20.49pp | 0.339 | -0.52pp | -0.019 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 30.08pp | 0.557 | -3.03pp | -0.045 |
| test_2023_2024 | 30.08pp | 0.557 | -3.03pp | -0.045 |
