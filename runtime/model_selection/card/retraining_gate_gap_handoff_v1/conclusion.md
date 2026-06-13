# Retraining Gate Gap Handoff V1

- Decision: `run_full_walkforward_cap10_joint_contract_bc_checkpoint_training`.
- Promotion: blocked; this card does not retain or promote a new actor.
- Retention gate: 0/12 replayable profiles pass complete WF + MaxDD/return alignment.
- SOXLBacktest reference: `soxl_backtest_policy_reimplementation_v1` YTD `132.077978%`; current-best alignment floor `213.143462%`.
- Best available next source: cap10 joint-contract BC/checkpoint path. wf_2022 selected MaxDD `18.455088%`, rel `36.419124pp`; wf_2023_2024 selected MaxDD `13.597475%`, rel `30.501116pp`.
- Critical stop rule: do not lengthen PPO when wf_2022 MaxDD drift vs BC exceeds `2pp`; observed PPO-selected drift was `18.820132pp`.
- Next retraining run must add wf_2020 coverage before any actor retention, then rerun fixed-split + full walk-forward + dashboard replay export.
