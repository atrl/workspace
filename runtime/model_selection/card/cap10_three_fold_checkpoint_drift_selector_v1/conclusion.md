## Conclusion

Decision: `three_fold_checkpoint_selector_partial_pass_no_promotion`.

Gate: relative return >= 30pp and MaxDD < 20% on `wf_2020`, `wf_2022`, and `wf_2023_2024`.

Selected candidates:
- wf_2020: `bc_initialization` (bc_initialization), rel -285.45pp, MaxDD 19.42%, loss 10.5150, gate False
- wf_2022: `bc_initialization` (bc_initialization), rel 47.13pp, MaxDD 18.66%, loss 0.0000, gate True
- wf_2023_2024: `bc_initialization` (bc_initialization), rel 38.71pp, MaxDD 10.40%, loss 0.0000, gate True

Risk note: selector output is diagnostic only. It retains no actor and does not override the full retained actor + fixed-split + walk-forward + YTD return-alignment promotion requirement.

Next action: `inspect_failed_fold_drawdown_or_reward_drift_before_full_retention_training`.

Artifacts: `checkpoint_candidate_metrics.csv`, `checkpoint_selection_by_fold.csv`, `drift_score_summary.json`, `selection.json`, `summary.json`.
