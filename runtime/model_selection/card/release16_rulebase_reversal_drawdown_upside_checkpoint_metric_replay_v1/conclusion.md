# Release16 Rulebase Reversal Drawdown/Upside Checkpoint Metric Replay

Decision: `release16_rulebase_reversal_drawdown_upside_checkpoint_metric_replay_partial_pass_no_promotion`.

Gate: relative return >= 30pp and MaxDD < 20% on `wf_2020`, `wf_2022`, and `wf_2023_2024`.

Selected candidates:
- wf_2020: `bc_initialization` (bc_initialization), rel -0.43pp, MaxDD 19.33%, loss 1.0142, gate False, vs SOXLBacktest return delta 28.51pp.
- wf_2022: `bc_initialization` (bc_initialization), rel 71.71pp, MaxDD 15.62%, loss 0.0000, gate True, vs SOXLBacktest return delta 1.17pp.
- wf_2023_2024: `bc_initialization` (bc_initialization), rel 2.99pp, MaxDD 13.62%, loss 0.9003, gate False, vs SOXLBacktest return delta -1.87pp.

Direct checkpoint delta vs prior rulebase reversal replay:
- wf_2020: return -5.51pp, relative -5.51pp, MaxDD -1.61pp, avg position +0.006.
- wf_2022: return +0.00pp, relative +0.00pp, MaxDD +0.00pp, avg position +0.000.
- wf_2023_2024: return -8.59pp, relative -8.59pp, MaxDD +1.87pp, avg position +0.073.

Metric basis: selected metrics use `official_daily_path_rebased_eval_window`; `12` candidate rows have raw-vs-daily relative mismatch >= 5pp in `metric_basis_reconciliation.csv`.

Risk note: selector output is diagnostic only. It retains no actor and does not override the full retained actor + fixed-split + walk-forward + YTD return-alignment promotion requirement.

Next action: `diagnose_remaining_rulebase_reversal_failed_folds_before_full_retention_training`.

Artifacts: `checkpoint_candidate_metrics.csv`, `checkpoint_selection_by_fold.csv`, `metric_basis_reconciliation.csv`, `official_daily_paths/*.csv`, `drift_score_summary.json`, `selection.json`, `summary.json`.
