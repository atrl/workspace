# Network Knowledge

## Retained Insight

Network changes are only justified when they address a specific behavioral bottleneck such as asset-specific behavior, transition classification, or temporal abstraction.

## Failed Pattern

Larger bundled structures, attention swaps, or MoE additions do not automatically repair drawdown or restart behavior and can hide whether the issue is policy, reward, execution, or data.

## Evidence Path

- `memory/2026-04-22-learned-mode-classifier-and-hybrid-experiments.md`
- `memory/2026-04-22-v10-attention-simplified-reward-failure.md`
- `docs/experiment-landscape-review.md`

## Reusable Method

State the bottleneck first, add the smallest architecture needed, and compare against the direct baseline with fixed splits and behavior diagnostics.

## Next Valid Experiment Type

For cross-ETF work, test asset-specific heads or volatility-scale conditioning rather than a single shared actor with one-hot identity alone.
