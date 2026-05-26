# Constrained Checkpoint Selector

## Decision

- decision: `constrained_checkpoint_selector_no_promotion`
- selected_candidate: `short_volshock_plain`
- selector_gate_pass: `False`
- promotion_allowed: `False`
- next_mechanism: `generate_reward_constrained_short_training_candidate`

## Candidate Ranking

| candidate | gate | loss | pass/windows | min rel | max dd | rel fails | dd fails |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short_volshock_plain | False | 322.43 | 0/2 | 5.02pp | 22.26% | 1 | 2 |
| short_volshock_weighted_obj025_cb099 | False | 325.69 | 0/2 | 8.82pp | 24.75% | 1 | 1 |
| base_multi_obj_plain | False | 604.18 | 0/2 | 33.76pp | 36.33% | 0 | 2 |

## Readout

- A full-gate candidate still requires replay review before promotion.
- If no candidate clears every window, the selector only identifies the closest diagnostic candidate.
