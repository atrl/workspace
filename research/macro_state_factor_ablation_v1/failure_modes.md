# Failure Modes

- promotion_decision: not_promoted_validation_selected_baseline
- failed_checks: ["direct_base_comparison"]
- revision_limit: Treasury curves and the broad-dollar mirror are current-vintage history, not revision-vintage archives.
- inflation_proxy_limit: 5Y5Y inflation is derived from Treasury par curves and is not FRED T5YIFR.
- selection_limit: feature groups are selected on validation only; outer-test results are report-only.
- interpretation_limit: macro state factors are model inputs, not standalone trading rules.
