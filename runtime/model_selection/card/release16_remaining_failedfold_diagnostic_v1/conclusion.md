# Release16 Remaining Failed-Fold Diagnostic

Decision: `release16_overlay_failure_is_state_transfer_and_representation_gap_no_promotion`.

Gate gaps:
- wf_2020: rel 11.42pp, return gap to +30pp relative 18.58pp, MaxDD 16.65%, overlay relative delta +26.45pp.
- wf_2023_2024: rel 3.41pp, return gap to +30pp relative 26.59pp, MaxDD 5.55%, overlay relative delta -1.30pp.

Exposure transfer:
- wf_2020: train target mean 0.364 vs selected test mean 0.230; target share >=50% 0.302 vs selected share >=50% 0.071.
- wf_2023_2024: train target mean 0.237 vs selected test mean 0.083; target share >=50% 0.300 vs selected share >=50% 0.000.

Missed low-position up days:
- wf_2020: 47 days, mean prev position 0.102, SOXL up-day sum 376.80pp, 50% floor counterfactual 150.03pp.
- wf_2023_2024: 66 days, mean prev position 0.070, SOXL up-day sum 490.29pp, 50% floor counterfactual 211.57pp.

State/representation notes:
- wf_2020: proxy trigger days 13 / 252, low-position up-day trigger overlap 5 / 47.
- wf_2023_2024: proxy trigger days 141 / 251, low-position up-day trigger overlap 41 / 66.
- wf_2020: BC all accuracy 0.738, label dominant share 0.352, predicted dominant share 0.457, selected-minus-prior avg pos +0.006.
- wf_2023_2024: BC all accuracy 0.709, label dominant share 0.462, predicted dominant share 0.573, selected-minus-prior avg pos -0.002.

Risk note: this diagnostic does not train or retain an actor. It uses test-window daily paths only for failure attribution, not for target generation.

Next action: `mine_failedfold_test_state_features_then_rebuild_representation_or_reward_objective_before_training`.

Artifacts: `fold_gate_gap_summary.csv`, `monthly_overlay_vs_prior_vs_reference.csv`, `missed_up_days_after_overlay.csv`, `missed_up_days_summary.csv`, `target_to_selected_exposure_gap.csv`, `test_state_trigger_overlap.csv`, `bc_representation_summary.csv`, `selection.json`, `summary.json`, `browser_dashboard_kimi_runbacktest.png`.
