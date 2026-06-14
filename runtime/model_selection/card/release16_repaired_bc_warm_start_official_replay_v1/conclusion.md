# release16_repaired_bc_warm_start_official_replay_v1

Decision: `repaired_bc_warm_start_official_replay_failed_no_promotion`.
Selected candidate: `release128_cb995_obj000`.
Next mechanism: `repair_replay_objective_with_drawdown_drift_and_trend_participation_before_ppo`.

No actor checkpoint is retained or promoted from this screen.

Key evidence:
- Selected 20% gate count: 0 / 3.
- Selected strict 15% gate count: 0 / 3.
- Selected min relative return: -25.0140 pp.
- Selected worst MaxDD: 34.3403%.
- Failed folds under 20% gate: ['wf_2020', 'wf_2022', 'wf_2023_2024'].

Artifacts:
- `warm_start_official_metrics.csv`
- `candidate_summary.csv`
- `selected_vs_baseline_delta.csv`
- `official_daily_paths/*.csv`
