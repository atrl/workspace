# fold_frontier_failure_attribution_v1

Decision: `fold_frontier_wf2020_timing_gap_blocks_retention`.

Fold gates passed by existing frontier: `2/3`. MaxDD gates: `3/3`. Relative gates: `2/3`. Return-floor gates: `2/3`.

- Existing candidates pass 2/3 folds as a combined frontier; no full retention training is allowed from this card.
- wf_2020 best MaxDD-safe candidate is bull_reentry_three_fold_checkpoint_metric_replay_v1:bc_initialization at 55.60% return and 15.49% MaxDD, leaving 14.01pp to the 69.60% floor.
- wf_2020 highest-return near-miss reaches 61.32% return with 20.92% MaxDD, so the next repair should target a sub-1pp drawdown breach instead of replacing the whole target with risk-cap labels.
- wf_2022 and wf_2023_2024 already have passing candidates in existing checkpoint replay evidence; they should be protected comparison folds in the next run.

Next action: `build_wf2020_near_miss_return_timing_repair_then_rerun_fold_aware_three_fold_replay`.

Promotion note: `promotion_allowed=false`, `actor_retention_allowed=false`; this card diagnoses the fold frontier and target-repair boundary only.
