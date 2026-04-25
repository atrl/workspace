---
title: balanced_rule_residual_v22
direction: model
status: candidate
---

# balanced_rule_residual_v22

## Hypothesis

See profile.yaml for configuration delta.

## Decision

Candidate. Details in runtime/model/reviews/ and docs/experiment-landscape-review.md.

## Expectations

# Expectations for balanced_rule_residual_v22

## Hypothesis

External VIX term-structure (VIX/VIX3M, VIX9D/VIX), VVIX, and SKEW carry
transition-window information that pure SOXL-derived features miss. Adding
them should improve recognition of:

- 2020-04 stress-release (SKEW / vvix_zscore normalising)
- 2022-07 contango flip (vix_term_slope crossing 0)
- 2023-01 early trend restart (short_ratio reversing)

## Deltas vs v19

- Feature schema grows from 59 to 64 columns (+5 VIX term/SKEW/VVIX features).
- Everything else (reward, position_cap, model, execution) identical to v19.
- Obs dim grows from 315 (5 x 59 + 15 + 5) to 340 (5 x 64 + 15 + 5). Existing
  v19 actors are NOT backward-compatible; v22 is a fresh training run.

## Must-haves

- [ ] Training completes under `SOXL_USE_VIX_TERM=1`.
- [ ] `feature_schema.signature` in summary differs from v19 (sanity check).
- [ ] `test_2022_max_dd` not worse than v19 by more than 3pp (no regression).

## Nice-to-haves

- [ ] `test_2020` return >= v19 (+5pp) — validate stress-release capture.
- [ ] `test_2023_2024` return >= v19 (+10pp) — validate trend-restart capture.
- [ ] `test_2022_avg_pos` <= v19 (regime_gate still works).

## Kill criteria

- If test_2022 avg_pos rises above 0.35 (gate breach), revert to v19.
- If overall `test_proxy` Sharpe drops below 0.80, revert.

## Comparison anchor

- `balanced_rule_residual_v19` (seed=12345, 59-feature).
- Slice report target:
  `runtime/model/reviews/balanced_rule_residual_v22_vs_v19.md`.
