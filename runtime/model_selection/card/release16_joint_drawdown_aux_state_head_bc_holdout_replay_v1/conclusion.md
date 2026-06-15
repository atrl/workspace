# release16_joint_drawdown_aux_state_head_bc_holdout_replay_v1

Decision: `release16_joint_drawdown_aux_state_head_bc_holdout_slice_failed_no_promotion`.

Full fold gates passed: `0/3`.
Holdout slice gates passed: `0/3`.
Strict drawdown gates passed: `1/3`.
Relative return gates passed: `1/3`.
Drift gates passed: `0/3`.

Min selected relative return: `-5.32pp`; worst selected MaxDD: `29.52%`.

Selected by fold:
- `wf_2020`: `risk_guarded_aux`, rel `-5.32pp`, MaxDD `20.93%`, holdout_slice_gate `False`, failed `all,is_rebound_cluster`.
- `wf_2022`: `drawdown_heavy_aux`, rel `58.11pp`, MaxDD `29.52%`, holdout_slice_gate `False`, failed `is_downtrend_state`.
- `wf_2023_2024`: `class_balanced_aux`, rel `29.77pp`, MaxDD `17.37%`, holdout_slice_gate `False`, failed `is_downtrend_state,drawdown_control_state`.

Risk note: this is a diagnostic BC holdout and official replay screen. It does not retain an actor and does not authorize PPO or production promotion.

Next action: `repair_action_head_sampling_or_aux_state_loss_before_official_replay`.
