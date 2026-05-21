# Budget-Usage Runtime Promotion Review

- Profile: `v19reg_v22_budget_usage_bull_guard_v1`
- Decision: `approved`
- Runtime model gate allowed: `True`
- Slow profile: `balanced_rule_residual_v19_reg`
- Fast profile: `balanced_rule_residual_v22`

## Checks

| check | passed | detail |
|---|---:|---|
| `required_windows_present` | `True` | all required walk-forward payloads are present |
| `bull_guard_enabled` | `True` | price-derived bull guard is enabled in all payloads |
| `all_windows_return_beats_fast` | `True` | candidate beats the fast v22 path return in every required window |
| `all_windows_maxdd_not_worse_than_fast` | `True` | candidate MaxDD is not worse than the fast v22 path in every required window |
| `wf_2022_defense_preserved` | `True` | 2022 defense remains within runtime promotion limits |

## Window Deltas

| window | delta_return | delta_max_dd | delta_avg_pos |
|---|---:|---:|---:|
| `wf_2020` | 36.85pp | -8.21pp | -0.003 |
| `wf_2022` | 23.42pp | -19.30pp | -0.044 |
| `wf_2023_2024` | 25.64pp | -3.03pp | -0.025 |

## Blocking Reasons

- none
