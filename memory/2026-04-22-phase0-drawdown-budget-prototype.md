---
date: '2026-04-22'
title: Phase 0 — Drawdown Budget Action Masking Prototype Results
kind: review
profiles:
- balanced_rule_residual_v19
- balanced_rule_residual_mode_v1
status: reference
tags:
- drawdown-budget
- action-masking
- phase0
- prototype
- approach-c
artifacts:
- runtime/post_train_review/drawdown_budget_sweep.json
- runtime/post_train_review/drawdown_budget_conditional.json
workpad: /Users/rziiiii/.kimi/plans/guy-gardner-black-hawk-silk.md
summary: 'Prototype of Approach C (drawdown-budget position masking) on fixed actors.
  Pure mechanical masking works but kills return. Regime-conditional masking (only
  active when regime_risk > 0.6) preserves return while modestly improving drawdown.
  mode_v1 already achieves 32% proxy MaxDD, making pure masking redundant. Core bottleneck
  is transition discrimination, not raw drawdown control.'
---

# Phase 0 — Drawdown Budget Action Masking Prototype Results

## Context

Plan: Breaking the 40% Drawdown Barrier. Approach C is the fastest-to-validate
mechanic: move drawdown control from reward penalties to hard action-space
constraints. Goal of Phase 0 is to measure the mechanical ceiling before
committing to a full training loop.

## Commands

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/prototype_drawdown_budget.py \
    --profile balanced_rule_residual_v19 --budgets no_budget,0.35,0.30,0.25

PYTHONPATH=. qlib_venv/bin/python scripts/prototype_drawdown_budget.py \
    --profile balanced_rule_residual_mode_v1 --budgets no_budget,0.35,0.30,0.25

PYTHONPATH=. qlib_venv/bin/python scripts/prototype_drawdown_budget_conditional.py
```

## Key Results

### v19 + unconditional budget masking

| Config | proxy return | proxy Sharpe | proxy MaxDD | 2022 MaxDD | 2023-24 return |
|---|---|---|---|---|---|
| baseline | 673.3% | 1.01 | 46.8% | 31.1% | 43.6% |
| budget 0.35 | 230.1% | 0.74 | 42.4% | 31.2% | 40.9% |
| **budget 0.30** | **353.0%** | **0.87** | **37.8%** | **27.7%** | **9.4%** |
| budget 0.25 | 177.9% | 0.74 | 36.8% | 26.6% | 14.9% |

**Interpretation:**
- budget 0.30 cuts proxy MaxDD by 9 pts (46.8% → 37.8%) but proxy return collapses by 320 pts.
- The 2023-2024 return crash (43.6% → 9.4%) shows the problem: in choppy markets, small drawdowns repeatedly trigger the budget, trapping the agent in low positions and causing it to miss rebounds.
- Pure mechanical masking is too blunt.

### mode_v1 + unconditional budget masking

| Config | proxy return | proxy Sharpe | proxy MaxDD |
|---|---|---|---|
| baseline | 343.6% | 1.03 | **32.0%** |
| budget 0.35 | 128.9% | 0.66 | 35.7% |
| budget 0.30 | 96.5% | 0.60 | 38.5% |
| budget 0.25 | 156.5% | 0.77 | 31.3% |

**Interpretation:**
- mode_v1 **already achieves 32.0% proxy MaxDD** — the target of 30% is almost reached.
- Adding extra masking on top of mode_v1 is either useless or harmful (return destruction with no drawdown improvement).
- **Conclusion:** The repo's drawdown problem is NOT "insufficiently tight hard constraints." mode_v1 proves the constraints already work.

### v19 + regime-conditional masking (only active when regime_risk >= threshold)

| Config | proxy return | proxy Sharpe | proxy MaxDD | 2022 MaxDD | 2023-24 return |
|---|---|---|---|---|---|
| baseline | 673.3% | 1.01 | 46.8% | 31.1% | 43.6% |
| rt0.4_b0.3 | 610.3% | 1.01 | 40.5% | 29.4% | 47.1% |
| rt0.5_b0.3 | 583.5% | 0.99 | 40.6% | 29.4% | 52.3% |
| **rt0.6_b0.3** | **669.5%** | **1.04** | **42.1%** | **29.4%** | **43.3%** |
| rt0.6_b0.25 | 620.3% | 1.02 | 42.3% | 28.9% | 43.7% |

**Interpretation:**
- rt0.6_b0.3 is the sweet spot: **return essentially unchanged** (669.5% vs 673.3%), Sharpe slightly up, MaxDD down 4.7 pts.
- 2023-2024 is protected because the mask only fires during genuine high-risk periods.
- This validates the core idea: **hard constraints should be conditional, not universal.**

## Decision

1. **Pure mechanical action masking (Approach C standalone) is rejected.** It compresses return too aggressively and is redundant on mode_v1.
2. **Regime-conditional masking is retained as a safety module.** It can be combined with Approach A as a hard floor during extreme stress (regime_risk > 0.6).
3. **The real problem is confirmed:** mode_v1 already gets to 32% proxy MaxDD. The gap to 30% is small. The much larger problem is **recovering the 330pp of return that mode_v1 gave up vs v19** without reopening the drawdown floodgates.
4. **Next step:** Approach A — replace the brittle rule-based mode classifier with a learned one that can discriminate transitions accurately. Better transition discrimination means the model can stay in `trend_hold` / `crash_rebuild` during genuine opportunities while still flipping to `flat_wait` during true stress.

## Retrieval tags

- drawdown-budget
- action-masking
- regime-conditional
- mode-v1
- v19
- transition-discrimination
