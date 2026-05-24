# recovery_reentry_equity_drawdown_guard_v1_wf_2020

Walk-forward anchor-confirmed recovery reentry with account-equity drawdown governor for wf_2020.

- Slow path: `balanced_rule_residual_v19_reg_wf_2020`
- Fast usage path: `balanced_rule_residual_v22_wf_2020`
- Composed path: `recovery_reentry_equity_drawdown_guard_v1_wf_2020`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_wf_2020 | 69.45% | 1.186 | 28.64% | 0.387 | 171 |
| balanced_rule_residual_v22_wf_2020 | 61.42% | 1.077 | 34.35% | 0.402 | 177 |
| recovery_reentry_equity_drawdown_guard_v1_wf_2020 | 42.68% | 0.953 | 24.37% | 0.343 | 40 |

## Slices

| slice | balanced_rule_residual_v19_reg_wf_2020 | balanced_rule_residual_v22_wf_2020 | recovery_reentry_equity_drawdown_guard_v1_wf_2020 |
| --- | --- | --- | --- |
| test_2020 | 69.45% / 1.186 / 28.64% / pos 0.387 / trades 171 | 61.42% / 1.077 / 34.35% / pos 0.402 / trades 177 | 42.68% / 0.953 / 24.37% / pos 0.343 / trades 40 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | -26.77pp | -0.233 | -4.27pp | -0.044 |
| test_2020 | -26.77pp | -0.233 | -4.27pp | -0.044 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | -18.74pp | -0.124 | -9.98pp | -0.058 |
| test_2020 | -18.74pp | -0.124 | -9.98pp | -0.058 |
