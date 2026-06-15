# release16_aux_state_loss_bc_holdout_slice_screen_v1

Decision: `release16_aux_state_loss_slice_failed_no_promotion`.

Full fold gates passed: `0/3`.
Holdout slice gates passed: `0/3`.
Strict drawdown gates passed: `1/3`.
Relative return gates passed: `1/3`.
Drift gates passed: `1/3`.
Trend preservation floor passed: `3/3`.

Min selected relative return: `0.25pp`; worst selected MaxDD: `27.42%`.

Selected by fold:
- `wf_2020`: `aux_release_drawdown_trend_floor`, rel `0.25pp`, MaxDD `27.42%`, trend_floor `True`, holdout_slice_gate `False`, failed `all,is_rebound_cluster`.
- `wf_2022`: `aux_release_drawdown_trend_floor`, rel `65.75pp`, MaxDD `21.50%`, trend_floor `True`, holdout_slice_gate `False`, failed `is_downtrend_state,drawdown_control_state`.
- `wf_2023_2024`: `aux_release_downtrend_balanced`, rel `27.38pp`, MaxDD `15.94%`, trend_floor `True`, holdout_slice_gate `False`, failed `is_downtrend_state`.

Risk note: this is a diagnostic BC holdout and official replay screen. It does not retain an actor and does not authorize PPO or production promotion.

Next action: `escalate_to_checkpoint_objective_loss_or_target_semantics_for_downtrend_drawdown_holdout_before_ppo`.
