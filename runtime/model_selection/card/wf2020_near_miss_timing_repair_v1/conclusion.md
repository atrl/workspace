# wf2020_near_miss_timing_repair_v1 Conclusion

Decision: `wf2020_timing_repair_requires_checkpoint_daily_path_export`.

The highest wf_2020 near-miss is `cap10_three_fold_checkpoint_metric_replay_v2:ppo_retained_iter_4` with return `61.32%`, MaxDD `20.92%`, return gap `8.28pp`, and MaxDD gap `0.92pp`.

No promotion is allowed from this card because the checkpoint near-miss daily path and actor artifacts are not persisted. The next step is to export the checkpoint daily paths first, then rerun timing overlays against those real paths.

Sidecars: `candidate_availability.csv`, `available_daily_path_metrics.csv`, `summary.json`, `selection.json`, `repair_plan.json`.
