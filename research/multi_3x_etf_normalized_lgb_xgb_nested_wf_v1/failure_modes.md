# Failure Modes

- Manual regime template selection is explicitly rejected by the overfit audit.
- Leave-one-asset-out reduces but does not eliminate overfit risk because all LETFs share US equity beta.
- Common coverage is clipped to the shortest local 3x ETF qlib store, so SOXL after common end is not evaluated here.
- Feature normalization is per-asset rolling z-score/percentile; stale or split-adjustment errors in source stores can still propagate.
- Current promotion decision: not_promoted_multi_3x_nested_wf_goal_or_overfit_audit_failed.
