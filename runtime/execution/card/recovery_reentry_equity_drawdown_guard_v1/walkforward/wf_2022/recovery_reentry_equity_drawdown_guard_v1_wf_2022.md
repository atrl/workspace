# recovery_reentry_equity_drawdown_guard_v1_wf_2022

Walk-forward anchor-confirmed recovery reentry with account-equity drawdown governor for wf_2022.

- Slow path: `balanced_rule_residual_v19_reg_wf_2022`
- Fast usage path: `balanced_rule_residual_v22_wf_2022`
- Composed path: `recovery_reentry_equity_drawdown_guard_v1_wf_2022`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_wf_2022 | -47.26% | -1.357 | 49.52% | 0.273 | 182 |
| balanced_rule_residual_v22_wf_2022 | -44.11% | -1.500 | 45.95% | 0.228 | 181 |
| recovery_reentry_equity_drawdown_guard_v1_wf_2022 | -26.36% | -1.247 | 28.25% | 0.155 | 34 |

## Slices

| slice | balanced_rule_residual_v19_reg_wf_2022 | balanced_rule_residual_v22_wf_2022 | recovery_reentry_equity_drawdown_guard_v1_wf_2022 |
| --- | --- | --- | --- |
| test_2022 | -47.26% / -1.357 / 49.52% / pos 0.273 / trades 182 | -44.11% / -1.500 / 45.95% / pos 0.228 / trades 181 | -26.36% / -1.247 / 28.25% / pos 0.155 / trades 34 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 20.90pp | 0.110 | -21.27pp | -0.119 |
| test_2022 | 20.90pp | 0.110 | -21.27pp | -0.119 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 17.75pp | 0.253 | -17.70pp | -0.073 |
| test_2022 | 17.75pp | 0.253 | -17.70pp | -0.073 |
