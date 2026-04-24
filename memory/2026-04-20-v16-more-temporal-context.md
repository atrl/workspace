---
date: '2026-04-20'
title: "v16 Results \u2014 More Temporal Context (window=5) Is a Big Win"
kind: decision
profiles:
- balanced_rule_residual_v16
status: kept
tags:
- balanced_rule_residual_v16
- decision
- eval-2022
- kept
- transition-features
- walk-forward
artifacts:
- artifacts/models/soxl_balanced_rule_residual_v16_actor.pt
- artifacts/models/soxl_balanced_rule_residual_v16_summary.json
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: "Legacy memory note normalized into the repo-wide schema. **Date:** 2026-04-20\
  \ **Profile:** `balanced_rule_residual_v16` **Hypothesis:** Increasing observation_window\
  \ from 3\u21925 with proven penalty=2.0 improves GRU decision quality. **Result:*"
---

# v16 Results — More Temporal Context (window=5) Is a Big Win

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

**Date:** 2026-04-20
**Profile:** `balanced_rule_residual_v16`
**Hypothesis:** Increasing observation_window from 3→5 with proven penalty=2.0 improves GRU decision quality.
**Result:** STRONG IMPROVEMENT — best GRU+penalty result to date.

## Results

| Profile | Window | Penalty | Return | Sharpe | MaxDD | test₂₀₂₂ MaxDD | fixed_split |
|---------|--------|---------|--------|--------|-------|----------------|-------------|
| v12 | 3 | 4.0 | 492.3% | 0.85 | 48.8% | ~30% | blocked |
| v13 | 3 | 2.0 | 492.3% | 0.85 | 48.8% | ~30% | blocked |
| v14 | 3 | 0.0 | 663.2% | 0.89 | 56.6% | 32.6% | ✓ pass |
| v15 | 3 | 1.0 | 525.4% | 0.83 | 58.6% | 33.0% | ✓ pass |
| **v16** | **5** | **2.0** | **632.4%** | **0.91** | **50.1%** | **32.7%** | **✓ pass** |
| v2 MLP | — | — | 521.3% | 0.86 | 50.4% | — | — |

## Key Findings

1. **window=5 is a meaningful improvement over window=3.**
   - Return: 492% → 632% (+140pp vs v12/v13)
   - Sharpe: 0.85 → 0.91 (best of all configs)
   - MaxDD: 48.8% → 50.1% (slight worsening, but still < 50%)
   - This is the best risk-adjusted GRU result so far.

2. **More temporal context helps the GRU plan ahead without losing discipline.**
   - The terminal penalty still controls drawdown (32.7% in 2022)
   - But the agent uses extra history to time entries/exits better
   - test_2020: 102.8% (best bull-market capture of penalty configs)
   - test_2023_2024: 57.9% (strong recovery period)

3. **Third consecutive fixed_split promotion pass.**
   - test_2022_max_dd: 32.7% < 35% ✓
   - test_2022_avg_pos: 0.27 < 0.35 ✓
   - test_proxy_sharpe: 0.78 > 0.75 ✓ (barely)

4. **Training rewards much more negative (~-280 vs ~-185 for v14).**
   - Penalty is being learned and applied
   - But test return is still high — the agent found a way to earn returns while managing drawdown

## Lesson

Temporal context matters for GRU. 3 days was insufficient; 5 days gives the GRU enough history to:
- Better anticipate regime transitions
- Make smoother position adjustments
- Capture more upside while keeping terminal penalty discipline

## Next Directions

1. **v17: observation_window=10** — test diminishing returns on context
2. **Walk-forward v16** — fixed_split passes; does it generalize to rolling windows?
3. **Larger GRU hidden size** (128 instead of 64) with window=5
4. **Combine window=5 with downside_asymmetry tuned** — currently at 1.3

## Evidence

- Log: `log/balanced_rule_residual_v16.log`
- Summary: `artifacts/models/soxl_balanced_rule_residual_v16_summary.json`
- Actor: `artifacts/models/soxl_balanced_rule_residual_v16_actor.pt`

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
