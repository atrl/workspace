# Release16 Failed-Fold State-Head Checkpoint BC Screen

Decision: `release16_failedfold_state_head_checkpoint_bc_screen_no_full_gate_no_promotion`.

Gate: relative return >= 30pp and MaxDD < 20% on `wf_2020`, `wf_2022`, and `wf_2023_2024`.

Selected candidates:
- wf_2020: `bc_initialization` (bc_initialization), rel 27.84pp, MaxDD 26.37%, loss 0.3907, metric False, slice False, drift True, candidate False, vs SOXLBacktest return delta 56.78pp.
- wf_2022: `bc_initialization` (bc_initialization), rel 72.36pp, MaxDD 15.49%, loss 0.0000, metric True, slice False, drift True, candidate False, vs SOXLBacktest return delta 1.82pp.
- wf_2023_2024: `bc_initialization` (bc_initialization), rel -16.01pp, MaxDD 30.56%, loss 2.0618, metric False, slice False, drift True, candidate False, vs SOXLBacktest return delta -20.88pp.

Direct checkpoint delta vs prior drawdown/upside replay:
- wf_2020: return +28.27pp, relative +28.27pp, MaxDD +7.05pp, avg position +0.173.
- wf_2022: return +0.65pp, relative +0.65pp, MaxDD -0.13pp, avg position +0.006.
- wf_2023_2024: return -19.00pp, relative -19.00pp, MaxDD +16.94pp, avg position +0.092.

Metric basis: selected metrics use `official_daily_path_rebased_eval_window`; `12` candidate rows have raw-vs-daily relative mismatch >= 5pp in `metric_basis_reconciliation.csv`.

Risk note: selector output is diagnostic only. It retains no actor and does not override the full retained actor + fixed-split + walk-forward + YTD return-alignment promotion requirement.

Next action: `revise_state_head_target_or_model_objective_before_retention_training`.

Artifacts: `checkpoint_candidate_metrics.csv`, `checkpoint_selection_by_fold.csv`, `metric_basis_reconciliation.csv`, `official_daily_paths/*.csv`, `drift_score_summary.json`, `selection.json`, `summary.json`.
