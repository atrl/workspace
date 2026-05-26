# State-Conditioned Objective Screen

## Decision

- decision: `no_objective_pass_candidate`
- selected_candidate: `base_seq_cap10_two_of_three`
- next_mechanism: `train_state_conditioned_multi_objective`

## Ranking

| candidate | pass | loss | wf pass | cluster pass | rel loss | dd loss | part loss | worst wf | worst cluster |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| base_seq_cap10_two_of_three | False | 216.92 | 1/2 | False | 120.49 | 52.32 | 44.10 | wf_2023_2024 | wf2024_poor63_rank1 |
| trend_q80_floor55_guard10 | False | 245.60 | 1/2 | False | 129.71 | 87.14 | 28.75 | wf_2023_2024 | wf2024_poor63_rank1 |
| trend_q70_floor45_guard10 | False | 301.12 | 0/2 | False | 140.58 | 132.07 | 28.48 | wf_2023_2024 | wf2024_poor63_rank1 |
| trend_q70_floor45_noeqdd | False | 301.12 | 0/2 | False | 140.58 | 132.07 | 28.48 | wf_2023_2024 | wf2024_poor63_rank1 |
| trend_q60_floor45_guard15 | False | 332.16 | 0/2 | False | 125.52 | 181.07 | 25.57 | wf_2023_2024 | wf2024_poor63_rank1 |
| trend_q70_floor55_guard15 | False | 400.33 | 0/2 | False | 138.73 | 236.34 | 25.25 | wf_2023_2024 | wf2024_poor63_rank1 |

## Readout

- A passing row is a promotion-review input, not an automatic runtime bind.
- If no row passes, treat the selected row as a diagnostic target and keep no runtime promotion.
- Required terms should be implemented in training-time reward/checkpoint selection before another path-level threshold search.
