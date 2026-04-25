---
title: balanced_rule_residual_refined_v3
direction: model
status: candidate
---

# balanced_rule_residual_refined_v3

## Motivation

Keep a concise runnable card for registry status `candidate`.

## Baseline

- `balanced_rule_residual_v19`

## Mechanism Delta

- Family: `balanced`

## Evidence Links

- Registry notes: `experiments/candidate/balanced_rule_residual_refined_v3/notes.md`

Long result narratives belong in `runtime/model/` or `runtime/`, not this card.

## Expectations

# balanced_rule_residual_refined_v3 Expectations

## Acceptance Criteria

- Compare against the registry baseline on fixed-split slices before any promotion decision.
- Preserve `test_2022` drawdown and average-position discipline unless this card explicitly states an accepted trade-off.

## Expected Trade-Off

- Any return improvement must be checked against proxy Sharpe, max drawdown, and cross-window behavior.

## Failure Mode

- Drop or archive if validation uplift comes with worse fixed-split risk or duplicated historical failure patterns.

## Comparison-Table Row Key

- `balanced_rule_residual_refined_v3`
