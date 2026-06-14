# wf2020_failed_fold_rally_gap_attribution_v1 conclusion

Decision: `wf2020_low_exposure_rally_gap_requires_training_safe_floor_target_no_promotion`.

Key evidence:
- Base replay `wf_2020` return `57.72%`, relative `-275.61pp`, MaxDD `17.28%`.
- Calibrated relative gap remains `11.89pp`; return-floor gap remains `11.89pp`.
- November 2020 SOXL price path return `60.49%`, but model average position was `0.098`.
- Low-exposure positive rally days after 2020-10-01: `32`.
- Diagnostic overlay pass count: `13`; best overlay: `floor=0.35, mom5_prev>=-0.02, start=2020-10-01, return=82.36%, MaxDD=17.28%`.

Risk notes:
- The overlay screen is post-hoc attribution on the official `wf_2020` daily path; it is not a promotion result.
- Any next target must express this as training-safe features/rules and rerun the full three-fold official replay.

Next action: `build_training_safe_low_exposure_rally_floor_target_then_rerun_official_three_fold_checkpoint_replay`.
