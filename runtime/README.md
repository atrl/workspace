# Runtime

`runtime/` holds generated evidence and operator-facing outputs. It is not the place for source-of-truth config.

Read this directory by index, not by browsing every file.

## Recommended Entry Points

- `runtime/post_train_review/`
  - human-readable comparisons after a train or checkpoint-review loop
- `runtime/generalization/`
  - cross-ETF replay and overlay comparisons
- `runtime/diagnosis/`
  - focused one-off diagnostics, logs, and feature forensics
- `runtime/soxl/wf_workers/`
  - persisted worker JSON for walk-forward subprocess runs

## Rules

- Treat files here as evidence or cached outputs, not canonical intent.
- When two files overlap, prefer the higher-level review markdown over raw logs or CSV.
- If a new workflow creates many raw files, also write one compact summary file that points to the important artifacts.
- Do not manually delete large runtime artifacts without checking retention and whether a memory note or workpad still references them.
