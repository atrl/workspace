# Reward / Checkpoint Screen

## Decision

- decision: `reward_checkpoint_screen_no_promotion`
- selected_candidate: `teacher_short_volshock_cap05_hit2`
- next_mechanism: `train_drawdown_drift_reward_or_checkpoint_recipe`

## Required Objective Terms

- `checkpoint_objective_loss`
- `drawdown_drift_penalty`
- `short_volshock_state_deleverage_consistency`
- `trend_preservation_floor`

## Candidate Ranking

| candidate | full | loss | pass/windows | min rel | max dd | failures |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| teacher_short_volshock_cap05_hit2 | False | 31.37 | 1/3 | 0.00pp | 15.11% | coverage_missing:1,pass:1,joint_tradeoff:1 |
| source_multi_obj_floor45_cap20_riskfirst | False | 197.49 | 0/3 | 0.00pp | 30.63% | drawdown_control:2,coverage_missing:1 |

## Readout

- This screen is a pre-PPO reward/checkpoint diagnostic, not runtime promotion.
- If no full-gate candidate exists, use the required terms as the next reward/checkpoint recipe.
