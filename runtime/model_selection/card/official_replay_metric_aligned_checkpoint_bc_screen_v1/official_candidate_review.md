# Official Replay Candidate Mechanism PK

| candidate | mechanism | full | robust | near miss | loss | min rel | max dd | failure |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| short_volshock_official_replay_screen_v1::short_volshock_cap05_hit2 | short_volshock_action_cap_overlay | False | False | True | 1.26 | 29.82pp | 15.11% | near_miss |
| short_volshock_sell_guard_official_replay_v1::short_volshock_cap05_hit2_no_guard | short_volshock_sell_guard_overlay | False | False | True | 1.26 | 29.82pp | 15.11% | near_miss |
| short_volshock_official_replay_screen_v1::short_volshock_cap05_hit1 | short_volshock_action_cap_overlay | False | False | False | 2.90 | 27.10pp | 13.83% | trend_capture |
| short_volshock_sell_guard_official_replay_v1::short_volshock_cap05_hit2_sell_guard_zero | short_volshock_sell_guard_overlay | False | False | False | 5.16 | 32.21pp | 15.52% | drawdown_control |
| short_volshock_sell_guard_official_replay_v1::short_volshock_cap05_hit2_sell_guard_1pct | short_volshock_sell_guard_overlay | False | False | False | 5.18 | 32.32pp | 15.52% | drawdown_control |
| short_volshock_official_replay_screen_v1::short_volshock_cap08_hit1 | short_volshock_action_cap_overlay | False | False | False | 11.59 | 23.62pp | 15.52% | joint_tradeoff |
| official_replay_risk_state_target_screen_v1::volstress_all_targets_cap010 | risk_state_target_replay | False | False | False | 29.11 | 2.00pp | 15.11% | trend_capture |
| official_replay_risk_state_target_screen_v1::volstress_all_targets_cap005 | risk_state_target_replay | False | False | False | 45.01 | -3.16pp | 16.18% | joint_tradeoff |
| sequence_trend_joint_warm_start_official_eval_v1::trend_plain_contract | sequence_trend_joint_bc | False | False | False | 73.77 | 8.22pp | 20.20% | joint_tradeoff |
| official_replay_risk_state_target_screen_v1::bc_initialization | risk_state_target_replay | False | False | False | 114.53 | 5.02pp | 22.26% | joint_tradeoff |
| official_replay_risk_state_target_screen_v1::risk_state_all_targets_cap010 | risk_state_target_replay | False | False | False | 114.53 | 5.02pp | 22.26% | joint_tradeoff |
| official_replay_risk_state_target_screen_v1::risk_state_risk_target_cap010 | risk_state_target_replay | False | False | False | 114.53 | 5.02pp | 22.26% | joint_tradeoff |
| short_volshock_bc_checkpoint_alignment_v1::short_volshock_plain | short_volshock_checkpoint_bc | False | False | False | 114.53 | 5.02pp | 22.26% | joint_tradeoff |
| short_volshock_bc_initialization_official_replay_v1::bc_initialization | short_volshock_bc_initialization | False | False | False | 114.53 | 5.02pp | 22.26% | joint_tradeoff |
| short_volshock_bc_checkpoint_alignment_v1::short_volshock_weighted_obj025_cb099 | short_volshock_checkpoint_bc | False | False | False | 118.64 | 8.82pp | 24.75% | joint_tradeoff |
| sequence_trend_joint_warm_start_official_eval_v1::base_plain_contract | sequence_trend_joint_bc | False | False | False | 142.00 | -0.14pp | 20.99% | joint_tradeoff |
| sequence_trend_joint_warm_start_official_eval_v1::trend_plus_slope_low_cap15_weighted | sequence_trend_joint_bc | False | False | False | 163.11 | 6.33pp | 28.94% | joint_tradeoff |
| short_volshock_official_replay_screen_v1::baseline_no_overlay | short_volshock_action_cap_overlay | False | False | False | 167.49 | 30.26pp | 30.63% | drawdown_control |
| short_volshock_sell_guard_official_replay_v1::baseline_no_overlay | short_volshock_sell_guard_overlay | False | False | False | 167.49 | 30.26pp | 30.63% | drawdown_control |
| state_conditioned_multi_objective_bc_screen_v1::multi_obj_floor45_cap20_riskfirst | state_conditioned_multi_objective_bc | False | False | False | 167.49 | 30.26pp | 30.63% | drawdown_control |
| state_conditioned_multi_objective_bc_screen_v1::multi_obj_floor65_cap20_weighted | state_conditioned_multi_objective_bc | False | False | False | 212.55 | 22.62pp | 31.57% | joint_tradeoff |
| state_conditioned_multi_objective_bc_screen_v1::multi_obj_floor55_cap25_balanced | state_conditioned_multi_objective_bc | False | False | False | 230.39 | 8.06pp | 31.81% | joint_tradeoff |
| short_volshock_bc_checkpoint_alignment_v1::base_multi_obj_plain | short_volshock_checkpoint_bc | False | False | False | 272.39 | 33.76pp | 36.33% | drawdown_control |

Near-miss candidates are diagnostic only. They require wider replay or stress testing before promotion.
