---
date: '2026-04-20'
title: "v15 Weak Penalty Results \u2014 Penalty=1.0 Is in a Bad Zone"
kind: decision
profiles:
- balanced_rule_residual_v15
status: reference
tags:
- balanced_rule_residual_v15
- decision
- eval-2022
- reference
- walk-forward
artifacts:
- artifacts/models/soxl_balanced_rule_residual_v15_actor.pt
- artifacts/models/soxl_balanced_rule_residual_v15_summary.json
workpad: memory/index.json
summary: Legacy memory note normalized into the repo-wide schema. **Date:** 2026-04-20
  **Profile:** `balanced_rule_residual_v15` **Hypothesis:** A weaker terminal penalty
  (1.0) might find a sweet spot between v14 aggression and v12/v13 conservatism.
---

# v15 Weak Penalty Results — Penalty=1.0 Is in a Bad Zone

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

**Date:** 2026-04-20
**Profile:** `balanced_rule_residual_v15`
**Hypothesis:** A weaker terminal penalty (1.0) might find a sweet spot between v14 aggression and v12/v13 conservatism.
**Result:** HYPOTHESIS REJECTED — penalty=1.0 produces the worst of both worlds.

## Results

| Profile | Penalty | Return | Sharpe | MaxDD | test₂₀₂₂ MaxDD | fixed_split |
|---------|---------|--------|--------|-------|----------------|-------------|
| v14 | 0.0 | 663.2% | 0.89 | 56.6% | 32.6% | ✓ pass |
| **v15** | **1.0** | **525.4%** | **0.83** | **58.6%** | **33.0%** | **✓ pass** |
| v12 | 4.0 | 492.3% | 0.85 | 48.8% | ~30% | (blocked)* |
| v13 | 2.0 | 492.3% | 0.85 | 48.8% | ~30% | (blocked)* |
| v2 MLP | — | 521.3% | 0.86 | 50.4% | — | — |

*Note: v12/v13 overall promotion blocked on other criteria; their fixed_split 2022 metrics likely pass.

## Key Findings

1. **Penalty=1.0 is NON-MONOTONIC and worse than both extremes.**
   - Lower return than v14 (525% vs 663%)
   - WORSE MaxDD than v14 (58.6% vs 56.6%)
   - WORSE MaxDD than v12/v13 (58.6% vs 48.8%)
   - This is the worst outcome of all four penalty settings.

2. **Training rewards confirm the penalty is firing.**
   - v14 training reward: ~-185 per episode
   - v15 training reward: ~-214 to -234 per episode
   - The penalty adds ~30-50 negative reward units per episode.

3. **The penalty at 1.0 may create a "confused" policy.**
   - Strong enough to distort the reward landscape
   - Not strong enough to reliably change behavior toward drawdown avoidance
   - Result: worse timing decisions, worse return, AND worse drawdown

4. **Both v14 and v15 pass fixed_split, but overall MaxDD is unacceptable.**
   - fixed_split only checks 2022 slice (32-33% MaxDD ✓)
   - Overall test MaxDD is 56-58% — worse than MLP baseline (50.4%)
   - The 2022 slice is well-behaved, but cross-boundary drawdowns are severe.

## Lesson

The terminal episode penalty has a **sharp threshold effect**, not a smooth dial:
- Below ~2.0: penalty is too weak to provide meaningful drawdown control
- At ~1.0: penalty adds noise without benefit — worst zone
- At 2.0-4.0: penalty engages effectively, GRU learns to exploit it

The effective operating range appears to be **≥2.0** for this seed and architecture.

## Next Directions

1. **Try penalty=0.5** to confirm the threshold hypothesis (if 0.5 ≈ v14, threshold is sharp)
2. **Increase observation_window to 5 or 10** with proven penalty=2.0 — more temporal context
3. **Try a different penalty formulation** — e.g., linear in drawdown instead of quadratic
4. **Walk-forward v14 or v15** — fixed_split passes for both; does it generalize?

## Evidence

- Log: `log/balanced_rule_residual_v15.log`
- Summary: `artifacts/models/soxl_balanced_rule_residual_v15_summary.json`
- Actor: `artifacts/models/soxl_balanced_rule_residual_v15_actor.pt`

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
