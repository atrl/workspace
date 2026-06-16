# Short Fold-Local Training Plan

## Decision

- decision: `short_foldlocal_training_plan_ready_no_promotion`
- training_allowed: `True`
- promotion_allowed: `False`
- next_mechanism: `run_short_foldlocal_ppo_screen`

## Fold Sources

| fold | source checkpoint | exists | max iters | stage iters |
| --- | --- | ---: | ---: | ---: |
| wf_2020 | `soxl_mode_conditioned_rebound_reward_train_v2_wf_2020_checkpoint_iter8_actor.pt` | True | 4 | 2 |
| wf_2022 | `soxl_mode_conditioned_rebound_reward_train_v2_wf_2022_checkpoint_iter4_actor.pt` | True | 4 | 2 |
| wf_2023_2024 | `soxl_mode_conditioned_rebound_reward_train_v2_wf_2023_2024_checkpoint_iter0_actor.pt` | True | 4 | 2 |

## Readout

- This plan prepares a diagnostic short fold-local training screen only.
- It does not permit promotion or actor retention; official walk-forward gates remain required.
