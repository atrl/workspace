# Model Selection Knowledge

## Retained Insight

Checkpoint choice often matters more than longer training. Several branches peaked early, and promotion should use fixed-split slices plus proxy metrics rather than headline return.

## Failed Pattern

Do not assume a branch is undertrained when validation improves but `test_2022` average position or max drawdown worsens.

## Use With Skills

- `post-train-backtest-review` for checkpoint comparison, fixed-split diagnosis, and promotion review.

## Evidence Path

- `runtime/model/knowledge.md`
- `runtime/model/knowledge.md`
- `runtime/model/`

## Reusable Method

Review `selection_history`, fixed-split `test_2020`, `test_2022`, `test_2023_2024`, and proxy MaxDD together. Prefer checkpoint review and fixed-actor counterfactuals before opening another broad retrain.

## Next Valid Experiment Type

A valid model-selection experiment changes checkpoint scoring or selection evidence, not reward coefficients or thresholds under a new name.

The v22/post-release-cap runtime-selection loop exposed a second selection
issue after the model and overlay gates passed: training `best_score` still
favored v19 even though corrected walk-forward evidence favored v22. Runtime
auto-selection now uses a complete walk-forward report when all standard
windows are present, scoring each window with the profile's selection weights
and relative outperformance as the return term. If no complete report exists it
falls back to the legacy summary `best_score`.

Current canonical runtime selection is `balanced_rule_residual_v22` with
`selective_release_recovery_post_release_cap_v1` explicitly approved for
auto-selection. The live walk-forward runtime selection scores are v22 `64.88`
versus v19 `46.17`.

## 2026-05-21 Continued-Training Diagnostic

`balanced_rule_residual_v22_continued_retention_v1` tested whether v22 needed
more PPO steps after checkpoint retention became available. It did not.
Continuing from the v22 actor retained iter `0`, `10`, and `20`, but iter `0`
remained best (`86.24` score) while iter `10` and iter `20` fell to `77.14` and
`69.26`. Proxy return increased, but risk widened: proxy MaxDD rose from v22
`54.52%` to `59.13%`, and `test_2022` worsened to `-51.24%` with MaxDD
`55.26%`, failing the fixed-split limit.

Conclusion: do not treat v22 as undertrained. Future model-selection work
should change checkpoint scoring, retained-evidence review, or a distinct
risk-control mechanism; a short continuation of v22 is not a valid promotion
path.

## 2026-05-21 Fixed-Path Budget-Usage Diagnostic

`v19reg_v22_budget_usage_v1` tested a model-selection / composition mechanism
instead of retraining: `balanced_rule_residual_v19_reg` defines the slow
risk-mode band, and `balanced_rule_residual_v22` is interpreted as fast usage
inside that band.

The first replay exposed stale baseline daily CSVs whose `test_2022` and
`test_2023_2024` positions were zero. Regenerate actor daily paths before
trusting fixed-path composition when slice coverage looks suspicious.

With fresh actor daily exports through `2026-05-19`, the composed path improved
versus fresh v22 on every fixed replay slice: proxy `2264.14% / 1.260 /
40.89% MaxDD`, `test_2022` `-21.20% / 31.09% MaxDD / 0.238 avg_pos`, and
`test_2023_2024` `244.80% / 1.606 / 30.67% MaxDD`.

Conclusion: keep as a diagnostic candidate and validate by walk-forward daily
composition before any runtime binding. This is a distinct mechanism from
continued PPO training, MoE routing, SOXX threshold tuning, or direct feature
retraining.

Walk-forward daily-path validation was mixed. `wf_2022` improved versus v22
from `-44.11% / 45.95% MaxDD / 0.228 avg_pos` to `-20.68% / 26.65% MaxDD /
0.184 avg_pos`, and `wf_2023_2024` improved from `32.56% / 33.21% MaxDD` to
`62.64% / 30.18% MaxDD`. But `wf_2020` return fell to `49.86%`, below both
v22 (`61.42%`) and v19_reg (`69.45%`) daily paths.

Conclusion update: diagnostic keep, no runtime binding. If this line continues,
the next mechanism must add a bull-participation guard or mode-band exception
and rerun all three walk-forward daily-path checks.
