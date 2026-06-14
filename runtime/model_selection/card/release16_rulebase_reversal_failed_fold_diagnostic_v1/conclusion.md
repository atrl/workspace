# Release16 Rulebase Reversal Failed-Fold Diagnostic

Decision: `rulebase_reversal_failed_fold_diagnostic_requires_target_revision_no_promotion`.

- wf_2020 failure is now drawdown-dominated: rel 5.08pp and MaxDD 20.93%.
- wf_2020 max drawdown segment 2020-09-02->2020-09-08 had price return -30.85% and avg new position 0.460 vs source overlay 0.191; the rulebase lift crossed the 20% drawdown ceiling.
- wf_2023_2024 failure is return-dominated: rel 11.58pp leaves a 18.42pp gap while MaxDD 11.75% stays below 20%.
- PPO continuation did not solve either failed fold; selected candidates remain BC initialization, so the next useful lever is target/objective revision rather than longer PPO from this target.
- Do not promote or retain any actor from this diagnostic.

Next action: `build_drawdown_capped_rulebase_reversal_and_wf2024_upside_participation_target_then_rerun_checkpoint_metric_replay`.

Artifacts: `failed_fold_daily_comparison.csv`, `failed_fold_drawdown_segments.csv`, `failed_fold_monthly_attribution.csv`, `failed_fold_top_gap_days.csv`, `target_pressure_summary.csv`, `revision_direction_matrix.csv`, `selection.json`, `summary.json`.
