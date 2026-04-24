---
date: '2026-04-20'
title: "v18 Results \u2014 v16 Config Is Seed-Sensitive on Return"
kind: decision
profiles:
- balanced_rule_residual_v18
status: kept
tags:
- balanced_rule_residual_v18
- decision
- eval-2022
- kept
- stress-regime
- walk-forward
artifacts:
- artifacts/models/soxl_balanced_rule_residual_v18_actor.pt
- artifacts/models/soxl_balanced_rule_residual_v18_summary.json
workpad: memory/index.json
summary: Legacy memory note normalized into the repo-wide schema. **Date:** 2026-04-20
  **Profile:** `balanced_rule_residual_v18` **Hypothesis:** v16's strong result (632.4%
  return) is reproducible with different seeds. **Result:** PARTIALLY REPRODUC
---

# v18 Results — v16 Config Is Seed-Sensitive on Return

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

**Date:** 2026-04-20
**Profile:** `balanced_rule_residual_v18`
**Hypothesis:** v16's strong result (632.4% return) is reproducible with different seeds.
**Result:** PARTIALLY REPRODUCIBLE — drawdown discipline is robust, but return is highly seed-dependent.

## Results

| Profile | Seed | Config | Return | Sharpe | MaxDD | test₂₀₂₂ MaxDD | fixed_split |
|---------|------|--------|--------|--------|-------|----------------|-------------|
| v16 | 20260420 | window=5, pen=2.0 | **632.4%** | **0.91** | **50.1%** | 32.7% | ✓ pass |
| **v18** | **42** | **window=5, pen=2.0** | **432.7%** | **0.81** | **50.8%** | **31.9%** | **✓ pass** |
| v12 | 20260420 | window=3, pen=4.0 | 492.3% | 0.85 | 48.8% | ~30% | blocked |

## Key Findings

1. **Drawdown discipline is SEED-ROBUST.**
   - v16: MaxDD 50.1%, test_2022 MaxDD 32.7%
   - v18: MaxDD 50.8%, test_2022 MaxDD 31.9%
   - Both pass fixed_split comfortably
   - The terminal penalty + GRU + window=5 consistently controls stress-window exposure

2. **Return is HIGHLY SEED-DEPENDENT (200pp swing!).**
   - v16 (seed=20260420): 632.4% return
   - v18 (seed=42): 432.7% return
   - Difference: -200pp for the exact same config
   - This means the random initialization strongly affects upside capture

3. **All 5 consecutive profiles pass fixed_split.**
   - v14: ✓ (penalty=0.0)
   - v15: ✓ (penalty=1.0)
   - v16: ✓ (penalty=2.0, window=5)
   - v17: ✓ (penalty=2.0, window=10)
   - v18: ✓ (penalty=2.0, window=5, seed=42)
   - The architecture is reliably producing 2022-compliant behavior

4. **test_proxy vs CLI backtest discrepancy.**
   - v16: test_proxy=413%, CLI=632%
   - v18: test_proxy=608%, CLI=433%
   - The two metrics are inversely correlated — test_proxy is NOT a reliable proxy for CLI backtest return

## Lesson

The GRU + terminal penalty + window=5 architecture is **reliably good at drawdown control** but **unreliable at return maximization**. The next optimization lever is **seed search** — finding the initialization that best captures upside while keeping the penalty-induced discipline.

## Next Directions

1. **Seed sweep with v16 config** — try seeds 12345, 20260411, 20260419, etc.
2. **Walk-forward v16** — test if the best seed generalizes to rolling windows
3. **Larger training budget** — more iterations or episodes might reduce seed sensitivity
4. **Ensemble multiple seeds** — combine policies from different seeds

## Evidence

- Log: `log/balanced_rule_residual_v18.log`
- Summary: `artifacts/models/soxl_balanced_rule_residual_v18_summary.json`
- Actor: `artifacts/models/soxl_balanced_rule_residual_v18_actor.pt`

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
