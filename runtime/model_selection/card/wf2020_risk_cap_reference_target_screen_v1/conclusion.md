# wf2020_risk_cap_reference_target_screen_v1

Decision: `wf2020_riskcap_reference_target_bc_lower_bound_ready_no_promotion`.

Selected variant: `riskcap_full_replacement`.
BC weighted accuracy `0.824`, min fold accuracy `0.647`, gate `True`.
Selected wf2020 avg target position `0.062` with `49` rows at target >=35% and `0` rows at target >=70%.

Key finding: the risk-cap full replacement is learnable and provides a non-leaking lower-bound target, but it is intentionally conservative and still needs file-backed replay before any retention decision. The return-preserving active/crash variants did not pass the BC screen, so they should not be used for PPO/checkpoint replay yet.

Promotion note: `promotion_allowed=false` and `actor_retention_allowed=false`; the next replay must still satisfy the calibrated wf2020 floor, wf2022/wf2023_2024 +30pp relative gates, MaxDD <20%, and YTD/common-window return floors.

Next action: `run_file_backed_wf2020_riskcap_full_bc_replay_before_any_retention_then_design_return_preserving_hybrid_target_if_ytd_gate_fails`.

Artifacts: `candidate_target_distribution.csv`, `buy_sell_feature_comparison.csv`, `bc_target_screen.csv`, `bc_fold_metrics.csv`, `bc_slice_metrics.csv`, `selection.json`, `summary.json`.
