# Release16 CB99 Rebound Checkpoint Screen

Decision: `release16_cb99_checkpoint_metric_screen_partial_pass_no_promotion`.

Gate: relative return >= 30pp and MaxDD < 20% on `wf_2020`, `wf_2022`, and `wf_2023_2024`.

Selected candidates:
- wf_2020: `bc_initialization` (bc_initialization), rel -290.27pp, MaxDD 14.96%, loss 10.6757, gate False, vs SOXLBacktest return delta 13.90pp.
- wf_2022: `bc_initialization` (bc_initialization), rel 63.70pp, MaxDD 15.62%, loss 0.0000, gate True, vs SOXLBacktest return delta 1.17pp.
- wf_2023_2024: `bc_initialization` (bc_initialization), rel 40.54pp, MaxDD 5.62%, loss 0.0000, gate True, vs SOXLBacktest return delta -0.15pp.

Risk note: selector output is diagnostic only. It retains no actor and does not override the full retained actor + fixed-split + walk-forward + YTD return-alignment promotion requirement.

Next action: `inspect_failed_release16_folds_before_full_retention_training`.

Artifacts: `checkpoint_candidate_metrics.csv`, `checkpoint_selection_by_fold.csv`, `official_daily_paths/*.csv`, `drift_score_summary.json`, `selection.json`, `summary.json`.
