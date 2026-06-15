# release16_wf2020_trend_target_bc_official_replay_v1

Decision: `release16_wf2020_trend_target_bc_slice_failed_no_promotion`.

Full fold gates passed: `0/3`.
Holdout slice gates passed: `1/3`.
Strict drawdown gates passed: `1/3`.
Relative return gates passed: `1/3`.
Drift gates passed: `1/3`.

Min selected relative return: `12.29pp`; worst selected MaxDD: `26.57%`.

Selected by fold:
- `wf_2020`: `risk_trend_balanced`, rel `12.29pp`, MaxDD `26.57%`, holdout_slice_gate `True`, failed ``.
- `wf_2022`: `risk_trend_balanced`, rel `65.27pp`, MaxDD `24.71%`, holdout_slice_gate `False`, failed `is_downtrend_state,drawdown_control_state`.
- `wf_2023_2024`: `risk_trend_balanced`, rel `19.04pp`, MaxDD `14.80%`, holdout_slice_gate `False`, failed `is_downtrend_state,drawdown_control_state`.

Risk note: this is a diagnostic BC holdout and official replay screen for an OOS-informed trend target. It does not retain an actor and does not authorize PPO or production promotion.

Next action: `repair_trend_slice_sampling_or_state_loss_before_official_replay`.

## Failure Attribution

- `wf_2020`: `joint_return_drawdown+drift_vs_runnable_reference`; next `drawdown_drift_penalty_or_checkpoint_scoring`; failed slices ``.
- `wf_2022`: `sparse_slice_fit+drawdown_control_gap+drift_vs_runnable_reference`; next `risk_state_or_drawdown_drift_checkpoint_objective`; failed slices `is_downtrend_state,drawdown_control_state`.
- `wf_2023_2024`: `sparse_slice_fit+underparticipation_relative_gap`; next `risk_state_or_drawdown_drift_checkpoint_objective`; failed slices `is_downtrend_state,drawdown_control_state`.

This blocks repeating the same trend-BC weight sweep as the next step.
