## Conclusion

Decision: `three_fold_checkpoint_metric_replay_partial_pass_no_promotion`.

Gate: relative return >= 30pp and MaxDD < 20% on `wf_2020`, `wf_2022`, and `wf_2023_2024`.

Selected candidates:
- wf_2020: `bc_initialization` (bc_initialization), rel -292.75pp, MaxDD 14.98%, loss 10.7584, gate False
- wf_2022: `bc_initialization` (bc_initialization), rel 65.86pp, MaxDD 18.80%, loss 0.0000, gate True
- wf_2023_2024: `ppo_retained_iter_4` (ppo_retained), rel 35.57pp, MaxDD 8.96%, loss 0.0040, gate True

Risk note: selector output is diagnostic only. It retains no actor and does not override the full retained actor + fixed-split + walk-forward + YTD return-alignment promotion requirement.

Next action: `inspect_failed_fold_drawdown_or_reward_drift_before_full_retention_training`.

Artifacts: `checkpoint_candidate_metrics.csv`, `checkpoint_selection_by_fold.csv`, `official_daily_paths/*.csv`, `drift_score_summary.json`, `selection.json`, `summary.json`.
