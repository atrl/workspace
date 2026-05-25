# State-Conditioned Objective Screen

## Decision

- decision: `no_objective_pass_candidate`
- selected_candidate: `selected_router_no_equity_dd`
- next_mechanism: `train_state_conditioned_multi_objective`

## Ranking

| candidate | pass | loss | wf pass | cluster pass | rel loss | dd loss | part loss | worst wf | worst cluster |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| selected_router_no_equity_dd | False | 967.40 | 0/4 | False | 689.74 | 267.01 | 10.64 | strict_oos | C003 |
| selected_router_dd12_22_cap25 | False | 984.29 | 0/4 | False | 710.76 | 262.60 | 10.94 | strict_oos | C003 |
| selected_router_dd10_20_cap20 | False | 1149.42 | 0/4 | False | 891.27 | 241.47 | 16.68 | strict_oos | C003 |
| selected_router_dd08_18_cap15 | False | 1391.04 | 0/4 | False | 1110.09 | 245.97 | 34.98 | strict_oos | C003 |

## Readout

- A passing row is a promotion-review input, not an automatic runtime bind.
- If no row passes, treat the selected row as a diagnostic target and keep no runtime promotion.
- Required terms should be implemented in training-time reward/checkpoint selection before another path-level threshold search.
