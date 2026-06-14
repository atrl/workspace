# Release16 Failed-Fold Overlay Checkpoint Screen

Decision: `release16_failedfold_overlay_checkpoint_metric_screen_partial_pass_no_promotion`.

Gate: relative return >= 30pp and MaxDD < 20% on `wf_2020`, `wf_2022`, and `wf_2023_2024`.

Selected candidates:
- wf_2020: `bc_initialization` (bc_initialization), rel 11.42pp, MaxDD 16.65%, loss 0.6193, gate False, vs SOXLBacktest return delta 40.36pp.
- wf_2022: `bc_initialization` (bc_initialization), rel 71.71pp, MaxDD 15.62%, loss 0.0000, gate True, vs SOXLBacktest return delta 1.17pp.
- wf_2023_2024: `bc_initialization` (bc_initialization), rel 3.41pp, MaxDD 5.55%, loss 0.8865, gate False, vs SOXLBacktest return delta -1.46pp.

Direct checkpoint delta vs prior release16 target:
- wf_2020: return +26.45pp, relative +26.45pp, MaxDD +1.69pp, avg position +0.006.
- wf_2022: return +0.00pp, relative +0.00pp, MaxDD +0.00pp, avg position +0.000.
- wf_2023_2024: return -1.30pp, relative -1.30pp, MaxDD -0.07pp, avg position -0.002.

Metric basis: selected metrics use `official_daily_path_rebased_eval_window`; `12` candidate rows have raw-vs-daily relative mismatch >= 5pp in `metric_basis_reconciliation.csv`.

Risk note: selector output is diagnostic only. It retains no actor and does not override the full retained actor + fixed-split + walk-forward + YTD return-alignment promotion requirement.

Next action: `diagnose_remaining_failed_release16_overlay_folds_before_full_retention_training`.

Artifacts: `checkpoint_candidate_metrics.csv`, `checkpoint_selection_by_fold.csv`, `metric_basis_reconciliation.csv`, `official_daily_paths/*.csv`, `drift_score_summary.json`, `selection.json`, `summary.json`.
