# CAP10 WF2020 Return-Preserving Target Revision V1

- Decision: `wf2020_return_preserving_target_revision_ready_no_promotion`.
- Promotion: blocked; this card retains no actor and starts no training.
- Source: `cap10_three_fold_checkpoint_metric_replay_v2` selected `wf_2020` `bc_initialization` with return `40.58%`, relative `-292.75pp`, and MaxDD `14.98%`.
- Root cause: `wf_2020` is under-participating, not over-drawing down. Drawdown headroom to the strict 20% gate is `5.02pp`.
- Calibrated target: `wf_2020` relative floor `-263.73pp`; current replay gap `29.02pp`.
- Baseline guardrail: pure `risk_cap_35_15_05` passes the calibrated `wf_2020` floor but has YTD return `18.31%`, below the SOXLBacktest YTD reference `132.08%` and current-best floor `213.14%`.
- Next action: `build_return_preserving_wf2020_bull_participation_target_then_rerun_three_fold_bc_checkpoint_metric_replay`.

Sidecars: `wf2020_return_preserving_gap.csv`, `target_revision_contract.json`, `selection.json`, `summary.json`.
