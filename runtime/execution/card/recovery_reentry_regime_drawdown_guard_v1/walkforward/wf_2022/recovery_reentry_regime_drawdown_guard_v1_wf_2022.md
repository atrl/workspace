# recovery_reentry_regime_drawdown_guard_v1_wf_2022

Walk-forward replay for notebook-selected recovery_reentry_regime_drawdown_guard_v1 on wf_2022.

- Slow path: `balanced_rule_residual_v19_reg_wf_2022`
- Fast usage path: `balanced_rule_residual_v22_wf_2022`
- Composed path: `recovery_reentry_regime_drawdown_guard_v1_wf_2022`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_wf_2022 | -47.26% | -1.357 | 49.52% | 0.273 | 182 |
| balanced_rule_residual_v22_wf_2022 | -44.11% | -1.500 | 45.95% | 0.228 | 181 |
| recovery_reentry_regime_drawdown_guard_v1_wf_2022 | -17.13% | -0.467 | 24.22% | 0.182 | 20 |

## Slices

| slice | balanced_rule_residual_v19_reg_wf_2022 | balanced_rule_residual_v22_wf_2022 | recovery_reentry_regime_drawdown_guard_v1_wf_2022 |
| --- | --- | --- | --- |
| test_2022 | -47.26% / -1.357 / 49.52% / pos 0.273 / trades 182 | -44.11% / -1.500 / 45.95% / pos 0.228 / trades 181 | -17.13% / -0.467 / 24.22% / pos 0.182 / trades 20 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 30.14pp | 0.890 | -25.30pp | -0.091 |
| test_2022 | 30.14pp | 0.890 | -25.30pp | -0.091 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 26.98pp | 1.033 | -21.73pp | -0.045 |
| test_2022 | 26.98pp | 1.033 | -21.73pp | -0.045 |
