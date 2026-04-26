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
