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
