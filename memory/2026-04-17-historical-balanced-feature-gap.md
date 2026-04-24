---
date: '2026-04-17'
title: '2026-04-17: historical-best `balanced` feature gap'
kind: failure
profiles:
- balanced
status: dropped
tags:
- balanced
- dropped
- failure
- stress-regime
- transition-features
artifacts:
- evidence/post_train_review/historical_balanced_feature_recall.md
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: 'Legacy memory note normalized into the repo-wide schema. Canonical `balanced`
  is already competent at: persistent stress defense mature trend capture Its main
  blind spot is: transition states especially stress-release, bear-market rebounds,'
---

# 2026-04-17: historical-best `balanced` feature gap

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Conclusion

- Canonical `balanced` is already competent at:
  - persistent stress defense
  - mature trend capture
- Its main blind spot is:
  - transition states
  - especially stress-release, bear-market rebounds, and early trend restarts

## Why this matters

- This explains why recent loops that only retuned reward weights failed:
  - the actor is seeing the wrong state abstraction, not just using the wrong penalty scalar

## Next feature direction

- Prioritize transition / persistence features before more reward tuning:
  - `regime_risk_delta_*`
  - `vol_stress_score_delta_*`
  - days-in-stress / days-below-anchor
  - rebound-from-low / days-since-low

## Evidence

- `evidence/post_train_review/historical_balanced_feature_recall.md`

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
