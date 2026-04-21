---
date: '2026-04-21'
title: 2026-04-21 v19 Dwell-Stress Merge Review
kind: review
profiles:
- soxl
status: active
tags:
- active
- eval-2022
- execution-overlay
- review
- soxl
- stress-regime
artifacts:
- runtime/post_train_review/v19_dwell_stress_merge_review.json
- runtime/post_train_review/v19_dwell_stress_merge_review.md
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: 'Follow-up to the `v19` fixed-actor overlay frontier review. The goal was
  to test one narrow merge only: keep the low-frequency persistence behavior of `dwell_hold_v2`,
  then add only the stress-side asymmetry that had helped `regime_route_v1'
---

# 2026-04-21 v19 Dwell-Stress Merge Review

## Context

Follow-up to the `v19` fixed-actor overlay frontier review. The goal was to test
one narrow merge only: keep the low-frequency persistence behavior of
`dwell_hold_v2`, then add only the stress-side asymmetry that had helped
`regime_route_v1` defend `2022`.

This was intentionally an execution-layer diagnostic, not a retraining loop.

## Evidence

- Workpad:
  - `codex_exec_plans/active/2026-04-10-soxl-agent-loop.md`
- Review artifacts:
  - `runtime/post_train_review/v19_dwell_stress_merge_review.md`
  - `runtime/post_train_review/v19_dwell_stress_merge_review.json`
- Command:
  - `PYTHONPATH=. qlib_venv/bin/python /tmp/v19_dwell_stress_merge_review.py`

## Results

Proxy-aligned ranking:

1. `regime_route_v1`
   - `659.1% / Sharpe 0.924 / MaxDD 48.85% / 464 trades`
2. `dwell_hold_v2`
   - `626.8% / Sharpe 0.883 / MaxDD 50.56% / 121 trades`
3. `base`
   - `567.4% / Sharpe 0.874 / MaxDD 49.28% / 332 trades`
4. `dwell_stress_merge_v1`
   - `451.7% / Sharpe 0.810 / MaxDD 60.60% / 358 trades`

Important slice comparison versus `dwell_hold_v2`:

- `dwell_stress_merge_v1` improved `test_2022`
  - return: `-24.6%` vs `-26.4%`
  - maxDD: `31.90%` vs `34.73%`
  - avg_pos: `0.267` vs `0.285`
- but it destroyed the intended dwell edge
  - proxy return: `451.7%` vs `626.8%`
  - proxy Sharpe: `0.810` vs `0.883`
  - proxy MaxDD: `60.60%` vs `50.56%`
  - trades: `358` vs `121`

## Decision

- Drop `dwell_stress_merge_v1`.
- Keep the frontier as two distinct overlay personalities:
  - `regime_route_v1` = crisis-control overlay
  - `dwell_hold_v2` = SOXL-specific low-churn persistence overlay
- Do not spend more loops on threshold-only merges between those two. That path
  is now evidence-backed repetition, not real exploration breadth.

## Next Action

If overlay work continues, it needs a genuinely new execution state or a new
information source. Otherwise the next meaningful search step should move back
to a different mechanism outside the current overlay parameter family.
