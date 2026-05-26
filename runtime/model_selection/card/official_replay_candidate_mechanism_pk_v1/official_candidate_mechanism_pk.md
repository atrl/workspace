# Official Replay Candidate Mechanism PK

| candidate | mechanism | full | robust | near miss | loss | min rel | max dd | failure |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| short_volshock_sell_guard::short_volshock_cap05_hit2_no_guard | action_cap_overlay | False | False | True | 1.26 | 29.82pp | 15.11% | near_miss |
| short_volshock_sell_guard::short_volshock_cap05_hit2_sell_guard_zero | action_cap_overlay | False | False | False | 5.16 | 32.21pp | 15.52% | drawdown_control |
| short_volshock_sell_guard::short_volshock_cap05_hit2_sell_guard_1pct | action_cap_overlay | False | False | False | 5.18 | 32.32pp | 15.52% | drawdown_control |
| risk_state_target_screen::volstress_all_targets_cap010 | hard_target_cap | False | False | False | 29.11 | 2.00pp | 15.11% | trend_capture |
| risk_state_target_screen::volstress_all_targets_cap005 | hard_target_cap | False | False | False | 45.01 | -3.16pp | 16.18% | joint_tradeoff |
| bc_initialization_official::bc_initialization | bc_warm_start_baseline | False | False | False | 114.53 | 5.02pp | 22.26% | joint_tradeoff |
| risk_state_target_screen::bc_initialization | hard_target_cap | False | False | False | 114.53 | 5.02pp | 22.26% | joint_tradeoff |
| risk_state_target_screen::risk_state_all_targets_cap010 | hard_target_cap | False | False | False | 114.53 | 5.02pp | 22.26% | joint_tradeoff |
| risk_state_target_screen::risk_state_risk_target_cap010 | hard_target_cap | False | False | False | 114.53 | 5.02pp | 22.26% | joint_tradeoff |
| short_volshock_sell_guard::baseline_no_overlay | action_cap_overlay | False | False | False | 167.49 | 30.26pp | 30.63% | drawdown_control |

Near-miss candidates are diagnostic only. They require wider replay or stress testing before promotion.
