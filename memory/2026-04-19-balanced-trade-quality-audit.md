---
date: 2026-04-19
title: 2026-04-19 balanced trade quality audit
kind: review
profiles:
- balanced
- balanced_trade_quality_audit
status: kept
tags:
- balanced
- balanced_trade_quality_audit
- kept
- review
artifacts:
- artifacts/models/soxl_balanced_summary.json
- runtime/model/diagnosis/phase1_phase2_combined.json
- runtime/model/reviews/balanced_trade_quality_audit.md
workpad: memory/index.json
summary: Applied the new post-train Kelly workflow to canonical `balanced`. The retained
  actor is still promotable at the portfolio level, but its edge is concentrated in
  larger and longer holds while `mid_3_10d`, `small_5_15%`, and most `micro_<5%` rebalances
  are weak or negative expectancy. Next optimization order stays fixed-actor trade
  compression before reward or architecture changes.
---

# 2026-04-19 balanced trade quality audit

## Context

The Kelly patch to the post-train review skill needed a concrete repo-owned tool and artifact, not just workflow text.

## Evidence

- Added repo-owned generator:
  - `scripts/trade_quality_audit.py`
- Validation:
  - `PYTHONPATH=. qlib_venv/bin/python -m py_compile scripts/trade_quality_audit.py`
  - `qlib_venv/bin/python scripts/trade_quality_audit.py --summary artifacts/models/soxl_balanced_summary.json --diagnosis runtime/model/diagnosis/phase1_phase2_combined.json --profile balanced --output runtime/model/reviews/balanced_trade_quality_audit.md`
  - `qlib_venv/bin/python scripts/check_harness.py`
- Canonical `balanced` remains the best retained baseline at the summary level:
  - `test_proxy = 468.90% / Sharpe 0.858 / MaxDD 50.18%`
  - fixed-split promotion gate passed
- Trade-quality breakdown:
  - `mid_3_10d = Kelly -0.3287 / Expectancy -3.02%`
  - `small_5_15% = Kelly -0.0624 / Expectancy -0.52%`
  - `micro_<5% = 109 trades / Kelly 0.0437 / Expectancy 0.38%`
- Friction interpretation:
  - `578` trade actions
  - rough friction estimate `57.8%`
  - removing `micro_<5%` would save about `21.8%`

## Conclusion

- The actor learned a useful portfolio-path policy, not a uniformly positive T policy.
- Negative Kelly in the weak buckets does not invalidate the actor.
- It does identify the next optimization order:
  - fixed-actor trade compression first
  - reward-side trade-quality shaping second
  - architecture changes later
