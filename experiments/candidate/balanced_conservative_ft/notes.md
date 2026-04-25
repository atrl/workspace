---
title: balanced_conservative_ft
direction: model
status: candidate
---

# balanced_conservative_ft

## Hypothesis

See profile.yaml for configuration delta.

## Decision

Candidate. Details in runtime/model/reviews/ and docs/experiment-landscape-review.md.

## Expectations

# balanced_conservative_ft Expectations

## Acceptance Criteria

- Compare against the registry baseline on fixed-split slices before any promotion decision.
- Preserve `test_2022` drawdown and average-position discipline unless this card explicitly states an accepted trade-off.

## Expected Trade-Off

- Any return improvement must be checked against proxy Sharpe, max drawdown, and cross-window behavior.

## Failure Mode

- Drop or archive if validation uplift comes with worse fixed-split risk or duplicated historical failure patterns.

## Comparison-Table Row Key

- `balanced_conservative_ft`
