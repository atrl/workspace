# State-Conditioned Objective Screen

## Decision

- decision: `no_objective_pass_candidate`
- selected_candidate: `low_floor_rebound_dd10`
- next_mechanism: `train_state_conditioned_multi_objective`

## Ranking

| candidate | pass | loss | wf pass | cluster pass | rel loss | dd loss | part loss | worst wf | worst cluster |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| low_floor_rebound_dd10 | False | 1137.83 | 0/12 | False | 937.58 | 168.76 | 31.48 | cap10_release_0.25__wf_2023_2024 | C003 |
| low_floor_rebound_dd08 | False | 1200.41 | 1/12 | False | 1051.55 | 107.90 | 40.96 | cap10_release_0.25__wf_2023_2024 | C003 |
| soxx_te_low_floor_dd10 | False | 1261.27 | 0/12 | False | 1079.74 | 139.28 | 42.24 | cap10_release_0.25__wf_2023_2024 | C003 |
| no_overlay_replay | False | 1282.89 | 4/12 | False | 1182.13 | 48.93 | 51.83 | cap10_release_0.15__wf_2023_2024 | C003 |
| soxx_strength_floor30_dd10 | False | 1352.21 | 0/12 | False | 1030.38 | 286.38 | 35.45 | cap10_release_0.10__wf_2023_2024 | C003 |
| permissive_floor35_dd12 | False | 1357.12 | 0/12 | False | 884.13 | 440.17 | 32.82 | cap10_release_0.10__wf_2023_2024 | C003 |
| trend_floor35_dd08 | False | 1401.11 | 1/12 | False | 1064.37 | 301.39 | 35.35 | cap10_release_0.15__wf_2023_2024 | C003 |
| strict_soxx_te45_dd15 | False | 1463.04 | 2/12 | False | 1196.96 | 219.16 | 46.92 | cap10_release_0.15__wf_2023_2024 | C003 |
| trend_floor45_dd08 | False | 1562.53 | 1/12 | False | 1096.13 | 418.99 | 47.42 | cap10_release_0.15__wf_2023_2024 | C003 |

## Readout

- A passing row is a promotion-review input, not an automatic runtime bind.
- If no row passes, treat the selected row as a diagnostic target and keep no runtime promotion.
- Required terms should be implemented in training-time reward/checkpoint selection before another path-level threshold search.
