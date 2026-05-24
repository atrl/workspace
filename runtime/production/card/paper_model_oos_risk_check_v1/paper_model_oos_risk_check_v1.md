# paper_model_oos_risk_check_v1

Current Futu paper-trading composer OOS risk check from strict post-train test data.

- Slow path: `balanced_rule_residual_v19_reg_current`
- Fast usage path: `balanced_rule_residual_v22_current`
- Composed path: `v19reg_v22_budget_usage_bull_guard_v1_current`

| variant | return | sharpe | max_dd | avg_pos | trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_rule_residual_v19_reg_current | 1667.81% | 1.151 | 43.53% | 0.450 | 1430 |
| balanced_rule_residual_v22_current | 1141.54% | 0.975 | 53.21% | 0.501 | 1437 |
| v19reg_v22_budget_usage_bull_guard_v1_current | 1542.89% | 1.109 | 41.59% | 0.462 | 199 |

## Slices

| slice | balanced_rule_residual_v19_reg_current | balanced_rule_residual_v22_current | v19reg_v22_budget_usage_bull_guard_v1_current |
| --- | --- | --- | --- |
| test_2020 | 128.61% / 1.697 / 28.64% / pos 0.500 / trades 231 | 108.04% / 1.363 / 44.57% / pos 0.561 / trades 231 | 154.60% / 1.840 / 25.27% / pos 0.536 / trades 30 |
| test_2022 | -31.32% / -0.606 / 43.01% / pos 0.328 / trades 235 | -41.73% / -0.738 / 52.84% / pos 0.399 / trades 240 | -21.57% / -0.440 / 31.10% / pos 0.248 / trades 26 |
| test_2023_2024 | 189.57% / 1.454 / 29.01% / pos 0.477 / trades 466 | 193.08% / 1.370 / 32.26% / pos 0.530 / trades 458 | 209.14% / 1.453 / 30.67% / pos 0.497 / trades 70 |

## Deltas

### composed_vs_slow

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | -124.91pp | -0.042 | -1.95pp | 0.012 |
| test_2020 | 25.99pp | 0.143 | -3.37pp | 0.036 |
| test_2022 | 9.75pp | 0.166 | -11.91pp | -0.079 |
| test_2023_2024 | 19.57pp | -0.001 | 1.65pp | 0.020 |

### composed_vs_fast

| window | delta_return | delta_sharpe | delta_max_dd | delta_avg_pos |
| --- | ---: | ---: | ---: | ---: |
| test_proxy | 401.36pp | 0.134 | -11.63pp | -0.039 |
| test_2020 | 46.55pp | 0.477 | -19.30pp | -0.025 |
| test_2022 | 20.17pp | 0.298 | -21.74pp | -0.151 |
| test_2023_2024 | 16.06pp | 0.083 | -1.59pp | -0.033 |
