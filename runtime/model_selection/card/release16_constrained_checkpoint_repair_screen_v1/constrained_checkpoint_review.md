# Constrained Checkpoint Selector

## Decision

- decision: `constrained_checkpoint_selector_no_promotion`
- selected_candidate: `bc_initialization`
- selector_gate_pass: `False`
- promotion_allowed: `False`
- next_mechanism: `train_reward_constrained_short_training_candidate_with_relative_floor_and_drawdown_gap_loss_then_official_three_fold_replay`

## Candidate Ranking

| candidate | gate | loss | pass/windows | min rel | max dd | rel fails | dd fails |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bc_initialization | False | 701.48 | 0/3 | -264.73pp | 26.83% | 1 | 3 |
| ppo_retained_iter_0_bc_equivalent | False | 701.48 | 0/3 | -264.73pp | 26.83% | 1 | 3 |
| ppo_retained_iter_4 | False | 877.42 | 0/3 | -262.99pp | 37.40% | 2 | 3 |
| ppo_selected_best | False | 877.42 | 0/3 | -262.99pp | 37.40% | 2 | 3 |

## Readout

- A full-gate candidate still requires replay review before promotion.
- If no candidate clears every window, the selector only identifies the closest diagnostic candidate.
