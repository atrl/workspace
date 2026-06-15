# Reward / Checkpoint Screen

## Decision

- decision: `reward_checkpoint_screen_no_promotion`
- selected_candidate: `bc_initialization`
- next_mechanism: `train_drawdown_drift_reward_or_checkpoint_recipe`

## Required Objective Terms

- `checkpoint_objective_loss`
- `drawdown_drift_penalty`
- `trend_preservation_floor`

## Candidate Ranking

| candidate | full | loss | pass/windows | min rel | max dd | failures |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| bc_initialization | False | 401.48 | 0/3 | -264.73pp | 26.83% | drawdown_control:2,joint_tradeoff:1 |
| ppo_retained_iter_0_bc_equivalent | False | 401.48 | 0/3 | -264.73pp | 26.83% | drawdown_control:2,joint_tradeoff:1 |
| ppo_retained_iter_4 | False | 668.20 | 0/3 | -262.99pp | 37.40% | joint_tradeoff:2,drawdown_control:1 |
| ppo_selected_best | False | 668.20 | 0/3 | -262.99pp | 37.40% | joint_tradeoff:2,drawdown_control:1 |

## Readout

- This screen is a pre-PPO reward/checkpoint diagnostic, not runtime promotion.
- If no full-gate candidate exists, use the required terms as the next reward/checkpoint recipe.
