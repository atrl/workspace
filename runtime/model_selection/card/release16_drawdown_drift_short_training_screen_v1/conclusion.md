## Conclusion

Decision: `release16_drawdown_drift_short_training_screen_mixed_no_promotion`.

Reward patch: `runtime/model_selection/card/release16_drawdown_drift_reward_recipe_v1/selected_reward_patch.json`.

Gate: relative return >= 30pp and MaxDD < 20% on `wf_2020`, `wf_2022`, and `wf_2023_2024`, plus non-negative short-training drift versus BC within configured tolerance.

Selected candidates:
- wf_2020: `ppo_retained_iter_4` (ppo_retained), rel -262.99pp, MaxDD 20.94%, avg_pos 0.388, gate False, daily `runtime/model_selection/card/release16_drawdown_drift_short_training_screen_v1/selected_daily_paths/wf_2020__ppo_retained_iter_4_daily_path.csv`
- wf_2022: `bc_initialization` (bc_initialization), rel 55.75pp, MaxDD 26.83%, avg_pos 0.163, gate False, daily `runtime/model_selection/card/release16_drawdown_drift_short_training_screen_v1/selected_daily_paths/wf_2022__bc_initialization_daily_path.csv`
- wf_2023_2024: `bc_initialization` (bc_initialization), rel 32.13pp, MaxDD 22.92%, avg_pos 0.186, gate False, daily `runtime/model_selection/card/release16_drawdown_drift_short_training_screen_v1/selected_daily_paths/wf_2023_2024__bc_initialization_daily_path.csv`

Risk note: this card does not retain or promote any actor. It only tests whether the selected drawdown-drift reward patch survives a short fold-local PPO screen after release16 BC initialization.

Next action: `revise_reward_patch_or_checkpoint_selection_before_full_training`.

Artifacts: `checkpoint_candidate_metrics.csv`, `checkpoint_selection_by_fold.csv`, `bc_slice_metrics.csv`, `ppo_selection_history.csv`, `selected_daily_paths/*.csv`, `selection.json`, `summary.json`.
