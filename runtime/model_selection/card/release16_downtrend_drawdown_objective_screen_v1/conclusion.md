# release16_downtrend_drawdown_objective_screen_v1

Decision: `release16_downtrend_drawdown_objective_drawdown_repaired_relative_gap_no_promotion`.

Full fold gates passed: `0/3`.
Holdout slice gates passed: `0/3`.
Strict drawdown gates passed: `3/3`.
Relative return gates passed: `2/3`.
Drift gates passed: `1/3`.

Min selected relative return: `-5.60pp`; worst selected MaxDD: `17.69%`.

Selected by fold:
- `wf_2020`: `dd_control_checkpoint`, rel `-5.60pp`, MaxDD `16.63%`, holdout_slice_gate `False`, failed `is_rebound_cluster`.
- `wf_2022`: `dd_control_weighted`, rel `72.98pp`, MaxDD `17.69%`, holdout_slice_gate `False`, failed `is_downtrend_state`.
- `wf_2023_2024`: `dd_control_overaction`, rel `33.37pp`, MaxDD `15.33%`, holdout_slice_gate `False`, failed `aux_state_head_release_state,is_rebound_cluster,is_downtrend_state,failedfold_state_head_target_state`.

Risk note: this is a diagnostic BC holdout and official replay screen. It does not retain an actor and does not authorize PPO or production promotion.

Next action: `repair_wf2020_trend_participation_without_losing_drawdown_control_then_rerun_official_replay`.
