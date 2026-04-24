# balanced_rule_residual_refined_v2 Expectations

## Acceptance Criteria

- Compare against the registry baseline on fixed-split slices before any promotion decision.
- Preserve `test_2022` drawdown and average-position discipline unless this card explicitly states an accepted trade-off.

## Expected Trade-Off

- Any return improvement must be checked against proxy Sharpe, max drawdown, and cross-window behavior.

## Failure Mode

- Drop or archive if validation uplift comes with worse fixed-split risk or duplicated historical failure patterns.

## Comparison-Table Row Key

- `balanced_rule_residual_refined_v2`
