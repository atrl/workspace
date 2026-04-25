---
title: balanced_rule_residual_mode_v1
direction: model
status: candidate
---

# balanced_rule_residual_mode_v1

## Motivation

Move beyond threshold/cooldown retuning to persistent high-level trading modes
that constrain the residual actor.

## Hypothesis

Multi-day behavior modes (trend_hold, defensive_hold, crash_rebuild, flat_wait)
should improve stress-period behavior more cleanly than threshold drift.

## Result

| Slice | Return | Sharpe | MaxDD | avg_pos |
|-------|--------|--------|-------|---------|
| test_proxy | 886.1% | 1.09 | 42.3% | 0.450 |
| test_2020 | 64.3% | — | 10.6% | — |
| test_2022 | -13.1% | — | 16.5% | 0.135 |
| test_2023_2024 | 43.0% | — | 20.1% | — |

## Decision

**Candidate.** Mode constraints improve stress/restart windows and proxy quality.
Cost: over-suppressed bull participation in test_2020 and validation.

## Evidence Links

- Landscape: `docs/experiment-landscape-review.md`
