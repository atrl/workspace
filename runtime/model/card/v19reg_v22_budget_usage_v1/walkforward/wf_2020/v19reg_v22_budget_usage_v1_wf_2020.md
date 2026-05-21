# v19reg_v22_budget_usage_v1_wf_2020

Walk-forward budget-usage composition: v19_reg wf_2020 sets the slow risk-mode band and v22 wf_2020 supplies usage inside that band.

- Slow path: `v19_reg_wf_2020`
- Fast usage path: `v22_wf_2020`
- Composed path: `v19reg_v22_budget_usage_v1_wf_2020`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| v19_reg_wf_2020 | 69.45% | 1.186 | 28.64% | 0.387 | 171 |
| v22_wf_2020 | 61.42% | 1.077 | 34.35% | 0.402 | 177 |
| v19reg_v22_budget_usage_v1_wf_2020 | 49.86% | 0.965 | 29.85% | 0.394 | 23 |

## Slices

| slice | v19_reg_wf_2020 | v22_wf_2020 | v19reg_v22_budget_usage_v1_wf_2020 |
| --- | --- | --- | --- |
| test_2020 | 69.45% / 1.186 / 28.64% / pos 0.387 / trades 171 | 61.42% / 1.077 / 34.35% / pos 0.402 / trades 177 | 49.86% / 0.965 / 29.85% / pos 0.394 / trades 23 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | -19.59pp | -0.221 | 1.21pp | 0.007 |
| test_2020 | -19.59pp | -0.221 | 1.21pp | 0.007 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | -11.56pp | -0.112 | -4.50pp | -0.007 |
| test_2020 | -11.56pp | -0.112 | -4.50pp | -0.007 |
