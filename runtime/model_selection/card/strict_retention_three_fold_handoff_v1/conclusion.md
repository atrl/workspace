# strict_retention_three_fold_handoff_v1 conclusion

Decision: `train_bull_trend_participation_with_drawdown_drift_stop_no_promotion`. Promotion remains blocked and no actor is retained.

Why: strict retention has `0/12` passing profiles. Required return is `213.143462%`, current best return is `224.361539%`, and every retained model candidate still must prove `wf_2020, wf_2022, wf_2023_2024` with MaxDD below `20.00%`.

SOXLBacktest status: `soxlbacktest_reference_qlib_aligned_not_promotable`. It is a qlib-aligned buy/sell reference baseline, not a promotable actor. The selected buy/sell reference is `legacy_default`; its legacy default has worst WF MaxDD `84.918053` and YTD return `132.077977`.

Three-fold checkpoint status: `three_fold_checkpoint_selector_partial_pass_no_promotion`. Failed folds: `['wf_2020']`. The primary failed fold is `wf_2020`.

Next action: use `soxlbacktest_reference_alignment, walkforward_relative_return_gap_penalty, checkpoint_objective_loss, wf2020_trend_participation_floor, drawdown_drift_penalty` as the next training/checkpoint objective. Do not continue the same PPO branch when MaxDD drift exceeds 2pp.
