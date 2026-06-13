## Conclusion

Decision: `checkpoint_selector_mixed_metric_source_no_promotion`.

The 2020 chart window and the checkpoint selector use different benchmark anchors:
- Calendar chart benchmark: 19.4640 -> 31.0987, return 59.78%.
- Warmup-aligned selector benchmark: 6.7520 -> 30.7713, return 355.74%.
- Gap: 295.96pp after 60 warmup rows.

The checkpoint benchmark itself matches the warmup-aligned price path, but the checkpoint relative-return field is mixed-source: qlib return minus checkpoint buy-and-hold differs from the saved relative return by up to 23.48pp.

This explains why Main Chart calendar returns can look inconsistent with model-selection relative-return gates. It does not make the failed `wf_2020` gate pass, and it does not promote a model.

Next action: Rerun checkpoint selection with backtest_profile bnh_return and relative_outperformance from the same qlib replay source before any new training.

Artifacts: `benchmark_alignment.csv`, `selection.json`, `summary.json`.
