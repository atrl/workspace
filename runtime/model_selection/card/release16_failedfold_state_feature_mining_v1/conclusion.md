# Release16 Failed-Fold State Feature Mining

Decision: `failedfold_state_head_features_ready_for_target_recipe_no_promotion`.

Failed-fold coverage:
- wf_2020: rows 252, feature rows 252, missed low-position up days 47, label-rule match 1.000, return 69.51%, MaxDD 16.65%.
- wf_2022: rows 250, feature rows 250, missed low-position up days 0, label-rule match 0.732, return -14.87%, MaxDD 15.62%.
- wf_2023_2024: rows 251, feature rows 251, missed low-position up days 66, label-rule match 1.000, return 3.05%, MaxDD 5.55%.

Feature-set gates:
- rulebase_reversal: test AUC min 0.864, test acc min 0.631, train-overlay AUC min 0.667, train-overlay acc min 0.576.
- compact_state_head: test AUC min 0.691, test acc min 0.540, train-overlay AUC min 0.662, train-overlay acc min 0.580.
- momentum_drawdown: test AUC min 0.625, test acc min 0.460, train-overlay AUC min 0.677, train-overlay acc min 0.605.
- transition_recovery: test AUC min 0.603, test acc min 0.679, train-overlay AUC min 0.609, train-overlay acc min 0.495.
- volatility_stress: test AUC min 0.542, test acc min 0.454, train-overlay AUC min 0.660, train-overlay acc min 0.437.
- soxx_relative_decay: test AUC min 0.376, test acc min 0.542, train-overlay AUC min 0.612, train-overlay acc min 0.565.

Top state-transfer aligned features:
- rulebase_reversal/RB_SELL_COMPOSITE: test effect 0.490, train effect 0.712, aligned=True.
- compact_state_head/mom_5: test effect 0.467, train effect 0.408, aligned=True.
- momentum_drawdown/mom_5: test effect 0.467, train effect 0.408, aligned=True.
- rulebase_reversal/RB_RSI6_OVERSOLD: test effect -0.400, train effect -0.360, aligned=True.
- compact_state_head/RB_TREND_REVERSAL: test effect 0.357, train effect 0.678, aligned=True.
- rulebase_reversal/RB_TREND_REVERSAL: test effect 0.357, train effect 0.678, aligned=True.
- rulebase_reversal/RB_CUMULATIVE_DROP_SEVERITY: test effect 0.329, train effect 0.713, aligned=True.
- momentum_drawdown/anchor_gap_180: test effect 0.325, train effect 0.696, aligned=True.

Best failed-test feature set: `rulebase_reversal`.
Best train-overlay feature set: `momentum_drawdown`.
Best transfer feature set: `rulebase_reversal`.
State feature recipe ready: `True`.
Target revision ready: `False`.

Risk note: this notebook uses failed test folds only for failure attribution and state-head feature mining; it does not generate eval-window training labels, train an actor, promote a model, or retain a checkpoint.

Next action: `build_rulebase_reversal_failedfold_state_head_target_recipe_then_checkpoint_screen`.

Artifacts: `missed_up_day_state_frame.csv`, `train_overlay_state_frame.csv`, `test_state_feature_set_scores.csv`, `test_state_fold_metrics.csv`, `test_state_feature_effects.csv`, `train_overlay_feature_set_scores.csv`, `train_overlay_fold_metrics.csv`, `train_overlay_feature_effects.csv`, `state_transfer_feature_alignment.csv`, `fold_signal_coverage.csv`, `candidate_feature_groups.json`, `selection.json`, `summary.json`.
