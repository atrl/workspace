# release16_action_head_position_loss_screen_v1

Decision: `release16_action_head_position_loss_partial_repair_downtrend_failed_no_promotion`.

Full fold gates passed: `0/3`.
Holdout slice gates passed: `1/3`.
Strict drawdown gates passed: `1/3`.
Relative return gates passed: `2/3`.
Drift gates passed: `1/3`.

Min selected relative return: `-13.66pp`; worst selected MaxDD: `28.34%`.

Selected by fold:
- `wf_2020`: `ordinal_mid_aux`, rel `-13.66pp`, MaxDD `25.58%`, holdout_slice_gate `True`, failed ``.
- `wf_2022`: `ordinal_high_aux`, rel `59.32pp`, MaxDD `28.34%`, holdout_slice_gate `False`, failed `is_downtrend_state`.
- `wf_2023_2024`: `ordinal_drawdown_aux`, rel `41.29pp`, MaxDD `13.88%`, holdout_slice_gate `False`, failed `is_downtrend_state,drawdown_control_state`.

Risk note: this is a diagnostic BC holdout and official replay screen. It does not retain an actor and does not authorize PPO or production promotion.

Next action: `add_explicit_downtrend_drawdown_aux_state_head_or_checkpoint_objective_before_official_replay`.
