# release16_joint_drawdown_release_target_contract_v1

Decision: `release16_joint_drawdown_aux_state_head_target_contract_ready_for_bc_screen_no_promotion`.

Selected contract variant: `aux_state_head_joint`.

Target contract gate: `True`. BC readiness gate: `True`.

Strict joint result: drawdown-control rows are added, but sparse release/state-head coverage remains constrained by guard-fold semantics.

Aux state-head joint result: guard-fold feature-hit rows are converted into a diagnostic release/rebound target candidate so the next BC holdout screen can test whether slice action fit improves without promoting an actor.

Risk note: this card does not retain an actor and does not authorize PPO. `wf_2020` path risk still requires downstream checkpoint/reward validation after slice target repair.

Next action: `run_release16_joint_drawdown_aux_state_head_bc_holdout_slice_screen_official_replay`.
