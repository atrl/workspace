# balanced_rule_residual_imitation_v4_budget

## Motivation

`balanced_rule_residual_imitation_v4` is the best participation-focused imitation branch, but it misses the stricter 30% MaxDD target by roughly 1-1.3pp on `test_2022` and `test_2023_2024`.

The drawdown-budget prototype showed that unconditional hard masking is too blunt, while regime-conditional masking at `regime_risk >= 0.60` and `budget=0.30` improved drawdown with little return drag on the v19 fixed actor.

## External Reference

- Internal open-source-style reference: `soxl_backtest` panic/risk gating mechanics.
- Existing harness evidence: `memory/2026-04-22-phase0-drawdown-budget-prototype.md`.

## Hypothesis

Keep the v4 participation reward behavior intact, then apply a production/runtime position-budget overlay only during high-risk regimes. This should reduce path drawdown without broadly suppressing rebound and trend participation.

## Target Slices / Windows

- `test_2022`: keep drawdown near or below 30% without increasing average position.
- `test_2023_2024`: preserve v4's 51.8% restart-window return as much as possible.
- `test_proxy`: keep Sharpe above the existing promotion floor.

## Acceptance Criteria

- `test_2022 MaxDD <= 30.0%`.
- `test_2022 avg_pos <= 0.35`.
- `test_proxy Sharpe >= 0.75`.
- `test_2023_2024 return` should not fall more than 5 percentage points from v4 unless MaxDD improves by at least 2 points.

## Expected Trade-Off

The overlay may trim some high-risk rebound exposure. That is acceptable only if the drawdown reduction is larger than the return drag in the target stress windows.

## Failure Mode

If the budget fires too often in choppy recovery regimes, it will recreate the unconditional-mask failure mode: low drawdown but weak restart capture.

## Comparison-Table Row Key

`balanced_rule_residual_imitation_v4_budget`
