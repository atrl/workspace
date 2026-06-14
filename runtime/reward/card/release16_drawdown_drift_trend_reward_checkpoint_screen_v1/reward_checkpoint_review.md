# Reward / Checkpoint Screen

## Decision

- decision: `reward_checkpoint_screen_no_promotion`
- selected_candidate: `release64_cb99_obj020`
- next_mechanism: `train_drawdown_drift_reward_or_checkpoint_recipe`

## Required Objective Terms

- `checkpoint_objective_loss`
- `drawdown_drift_penalty`
- `trend_preservation_floor`

## Candidate Ranking

| candidate | full | loss | pass/windows | min rel | max dd | failures |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| release64_cb99_obj020 | False | 145.35 | 1/3 | -2.03pp | 26.71% | joint_tradeoff:1,drawdown_control:1,pass:1 |
| release128_cb995_obj000 | False | 315.65 | 0/3 | -25.01pp | 34.34% | joint_tradeoff:1,drawdown_control:1,underparticipation:1 |
| current_readiness_baseline | False | 318.78 | 0/3 | -41.60pp | 34.73% | joint_tradeoff:1,drawdown_control:1,underparticipation:1 |

## Readout

- This screen is a pre-PPO reward/checkpoint diagnostic, not runtime promotion.
- If no full-gate candidate exists, use the required terms as the next reward/checkpoint recipe.
