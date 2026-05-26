# Reward / Checkpoint Screen

## Decision

- decision: `reward_checkpoint_screen_no_promotion`
- selected_candidate: `short_volshock_plain`
- next_mechanism: `train_drawdown_drift_reward_or_checkpoint_recipe`

## Required Objective Terms

- `checkpoint_objective_loss`
- `drawdown_drift_penalty`
- `short_volshock_state_deleverage_consistency`
- `trend_preservation_floor`

## Candidate Ranking

| candidate | full | loss | pass/windows | min rel | max dd | failures |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| short_volshock_plain | False | 121.72 | 0/2 | 5.02pp | 22.26% | drawdown_control:1,joint_tradeoff:1 |
| short_volshock_weighted_obj025_cb099 | False | 124.87 | 0/2 | 8.82pp | 24.75% | drawdown_control:1,selection_bias:1 |
| base_multi_obj_plain | False | 272.39 | 0/2 | 33.76pp | 36.33% | drawdown_control:2 |

## Readout

- This screen is a pre-PPO reward/checkpoint diagnostic, not runtime promotion.
- If no full-gate candidate exists, use the required terms as the next reward/checkpoint recipe.
