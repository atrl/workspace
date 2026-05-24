# recovery_reentry_equity_drawdown_guard_v1_wf_2023_2024

Walk-forward anchor-confirmed recovery reentry with account-equity drawdown governor for wf_2023_2024.

- Slow path: `balanced_rule_residual_v19_reg_wf_2023_2024`
- Fast usage path: `balanced_rule_residual_v22_wf_2023_2024`
- Composed path: `recovery_reentry_equity_drawdown_guard_v1_wf_2023_2024`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_wf_2023_2024 | 42.15% | 1.084 | 30.70% | 0.335 | 180 |
| balanced_rule_residual_v22_wf_2023_2024 | 32.56% | 0.866 | 33.21% | 0.362 | 178 |
| recovery_reentry_equity_drawdown_guard_v1_wf_2023_2024 | 47.91% | 1.245 | 26.56% | 0.284 | 52 |

## Slices

| slice | balanced_rule_residual_v19_reg_wf_2023_2024 | balanced_rule_residual_v22_wf_2023_2024 | recovery_reentry_equity_drawdown_guard_v1_wf_2023_2024 |
| --- | --- | --- | --- |
| test_2023_2024 | 42.15% / 1.084 / 30.70% / pos 0.335 / trades 180 | 32.56% / 0.866 / 33.21% / pos 0.362 / trades 178 | 47.91% / 1.245 / 26.56% / pos 0.284 / trades 52 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 5.76pp | 0.161 | -4.14pp | -0.051 |
| test_2023_2024 | 5.76pp | 0.161 | -4.14pp | -0.051 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 15.36pp | 0.379 | -6.64pp | -0.077 |
| test_2023_2024 | 15.36pp | 0.379 | -6.64pp | -0.077 |
