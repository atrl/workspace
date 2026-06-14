# Reward / Checkpoint Screen

## Decision

- decision: `reward_checkpoint_screen_no_promotion`
- selected_candidate: `release16_bull_reentry_participation_objective_replay_v1::bull65_hold3`
- next_mechanism: `train_drawdown_drift_reward_or_checkpoint_recipe`

## Required Objective Terms

- `checkpoint_objective_loss`
- `drawdown_drift_penalty`
- `trend_preservation_floor`

## Candidate Ranking

| candidate | full | loss | pass/windows | min rel | max dd | failures |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| release16_bull_reentry_participation_objective_replay_v1::bull65_hold3 | False | 34.48 | 1/3 | 14.30pp | 21.11% | joint_tradeoff:1,pass:1,underparticipation:1 |
| release16_bull_reentry_participation_objective_replay_v1::bull70_hold5 | False | 45.72 | 1/3 | 11.95pp | 20.94% | joint_tradeoff:1,pass:1,underparticipation:1 |
| release16_drawdown_upside_checkpoint_replay_v1::bc_initialization | False | 65.94 | 1/3 | -0.43pp | 19.33% | underparticipation:2,passed:1 |
| release16_drawdown_upside_checkpoint_replay_v1::ppo_retained_iter_0_bc_equivalent | False | 65.94 | 1/3 | -0.43pp | 19.33% | underparticipation:2,passed:1 |
| release16_state_head_post_fullprobe_objective_screen_v1::checkpoint_bc_initialization | False | 65.94 | 1/3 | -0.43pp | 19.33% | underparticipation:2,passed:1 |
| release16_state_head_post_fullprobe_objective_screen_v1::checkpoint_ppo_retained_iter_0_bc_equivalent | False | 65.94 | 1/3 | -0.43pp | 19.33% | underparticipation:2,passed:1 |
| soxlbacktest_qlib_walkforward_baseline_v1::wf_pass_ytd18_15_15_02 | False | 98.53 | 1/3 | -30.62pp | 18.51% | underparticipation:2,passed:1 |
| release16_bull_reentry_participation_objective_replay_v1::release64_current_target | False | 103.08 | 0/3 | 14.30pp | 26.86% | joint_tradeoff:1,drawdown_control:1,underparticipation:1 |
| release_gate_bc_official_replay_v1::release64_no_objective | False | 103.08 | 0/3 | 14.30pp | 26.86% | joint_tradeoff:1,drawdown_control:1,underparticipation:1 |
| soxlbacktest_qlib_walkforward_baseline_v1::low_risk_15_05_02 | False | 104.47 | 1/3 | -32.13pp | 19.46% | underparticipation:2,passed:1 |
| release_gate_bc_official_replay_v1::baseline_current | False | 111.28 | 0/3 | 7.19pp | 24.83% | joint_tradeoff:1,near_miss_drawdown:1,underparticipation:1 |
| soxlbacktest_qlib_walkforward_baseline_v1::low_risk_20_08_02 | False | 140.80 | 0/3 | -24.17pp | 24.73% | underparticipation:2,drawdown_control:1 |
| release16_state_head_post_fullprobe_objective_screen_v1::full_actor_probe | False | 150.63 | 1/3 | -22.72pp | 20.91% | underparticipation:1,pass:1,joint_tradeoff:1 |
| release16_drawdown_upside_checkpoint_replay_v1::ppo_retained_iter_4 | False | 167.51 | 1/3 | -13.71pp | 25.59% | joint_tradeoff:1,pass:1,underparticipation:1 |
| release16_drawdown_upside_checkpoint_replay_v1::ppo_selected_best | False | 167.51 | 1/3 | -13.71pp | 25.59% | joint_tradeoff:1,pass:1,underparticipation:1 |
| release16_state_head_post_fullprobe_objective_screen_v1::checkpoint_ppo_retained_iter_4 | False | 167.51 | 1/3 | -13.71pp | 25.59% | joint_tradeoff:1,pass:1,underparticipation:1 |
| release16_state_head_post_fullprobe_objective_screen_v1::checkpoint_ppo_selected_best | False | 167.51 | 1/3 | -13.71pp | 25.59% | joint_tradeoff:1,pass:1,underparticipation:1 |
| release16_bull_reentry_participation_objective_replay_v1::bull55_hold3 | False | 214.03 | 0/3 | 14.30pp | 34.11% | joint_tradeoff:1,drawdown_control:1,underparticipation:1 |
| soxlbacktest_qlib_walkforward_baseline_v1::risk_cap_35_15_05 | False | 366.39 | 0/3 | 7.01pp | 43.27% | underparticipation:2,drawdown_control:1 |
| soxlbacktest_qlib_walkforward_baseline_v1::legacy_default | False | 1728.54 | 0/3 | 5.54pp | 84.92% | joint_tradeoff:2,drawdown_control:1 |

## Readout

- This screen is a pre-PPO reward/checkpoint diagnostic, not runtime promotion.
- If no full-gate candidate exists, use the required terms as the next reward/checkpoint recipe.
