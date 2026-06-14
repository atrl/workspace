# Release16 Rulebase Reversal Checkpoint Metric Replay

Decision: `release16_rulebase_reversal_checkpoint_metric_replay_partial_pass_no_promotion`.

Gate: relative return >= 30pp and MaxDD < 20% on `wf_2020`, `wf_2022`, and `wf_2023_2024`.

Selected candidates:
- wf_2020: `bc_initialization` (bc_initialization), rel 5.08pp, MaxDD 20.93%, loss 0.8773, gate False, vs SOXLBacktest return delta 34.02pp.
- wf_2022: `bc_initialization` (bc_initialization), rel 71.71pp, MaxDD 15.62%, loss 0.0000, gate True, vs SOXLBacktest return delta 1.17pp.
- wf_2023_2024: `bc_initialization` (bc_initialization), rel 11.58pp, MaxDD 11.75%, loss 0.6140, gate False, vs SOXLBacktest return delta 6.72pp.

Direct checkpoint delta vs prior release16 target:
- wf_2020: return +20.11pp, relative +20.11pp, MaxDD +5.97pp, avg position +0.072.
- wf_2022: return +0.00pp, relative +0.00pp, MaxDD +0.00pp, avg position +0.000.
- wf_2023_2024: return +6.87pp, relative +6.87pp, MaxDD +6.13pp, avg position +0.059.

Metric basis: selected metrics use `official_daily_path_rebased_eval_window`; `12` candidate rows have raw-vs-daily relative mismatch >= 5pp in `metric_basis_reconciliation.csv`.

Risk note: selector output is diagnostic only. It retains no actor and does not override the full retained actor + fixed-split + walk-forward + YTD return-alignment promotion requirement.

Next action: `diagnose_remaining_rulebase_reversal_failed_folds_before_full_retention_training`.

Artifacts: `checkpoint_candidate_metrics.csv`, `checkpoint_selection_by_fold.csv`, `metric_basis_reconciliation.csv`, `official_daily_paths/*.csv`, `drift_score_summary.json`, `selection.json`, `summary.json`.
