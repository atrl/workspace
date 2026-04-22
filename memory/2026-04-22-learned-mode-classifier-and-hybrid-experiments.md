---
date: '2026-04-22'
title: Learned Mode Classifier + Hybrid Experiments — Full Results
type: review
profiles:
- balanced_rule_residual_v19
- balanced_rule_residual_mode_v1
- balanced_rule_residual_learned_mode_v1
status: reference
tags:
- learned-mode-router
- mode-classifier
- drawdown-budget
- actor-ensemble
- approach-a
- approach-c
- hybrid
- phase1
- transition-discrimination
- fundamental-frontier
workpad: /Users/rziiiii/.kimi/plans/guy-gardner-black-hawk-silk.md
summary: 'Comprehensive evaluation of three architectural approaches to break the 40% drawdown barrier. Learned mode classifier achieves 38.2% MaxDD on fixed actor but training adapts away from it. Regime-conditional budget masking preserves return while modestly improving drawdown. Actor ensemble fails. No configuration achieves 30% proxy MaxDD without large return sacrifice. The frontier appears to be ~35-42% for this leverage level and asset class.'
---

# Learned Mode Classifier + Hybrid Experiments — Full Results

## Context

Plan: Breaking the 40% Drawdown Barrier. Three approaches tested:
- **A:** Learned hierarchical risk budget (learned mode classifier replacing rule-based)
- **B:** Supervised MoE gate with tail-risk auxiliary task (not tested — deprioritized)
- **C:** Dynamic action masking / drawdown budget (tested as overlay and standalone)

## Commands

```bash
# Phase 0: drawdown budget prototypes
PYTHONPATH=. qlib_venv/bin/python scripts/prototype_drawdown_budget.py --profile balanced_rule_residual_v19 --budgets no_budget,0.35,0.30,0.25
PYTHONPATH=. qlib_venv/bin/python scripts/prototype_drawdown_budget.py --profile balanced_rule_residual_mode_v1 --budgets no_budget,0.35,0.30,0.25
PYTHONPATH=. qlib_venv/bin/python scripts/prototype_drawdown_budget_conditional.py

# Phase 0b: regime-conditional budget on v19
PYTHONPATH=. qlib_venv/bin/python scripts/prototype_drawdown_budget_conditional.py

# Phase 1: fit learned mode classifier
PYTHONPATH=. qlib_venv/bin/python scripts/fit_learned_mode_classifier.py --epochs 80 --seed 42

# Phase 1: compare mode routers on fixed actor
PYTHONPATH=. qlib_venv/bin/python scripts/compare_mode_routers.py --actor balanced_rule_residual_v19

# Phase 1: train actor under learned router
PYTHONPATH=. qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_rule_residual_learned_mode_v1 --skip-data-refresh --seed 12345

# Phase 1b: hybrid learned + budget on fixed actor
PYTHONPATH=. qlib_venv/bin/python scripts/prototype_hybrid_approach.py

# Phase 1c: actor ensemble (v19 + mode_v1 switching)
PYTHONPATH=. qlib_venv/bin/python scripts/prototype_actor_ensemble.py
```

## Master Results Table

| Configuration | proxy return | proxy Sharpe | proxy MaxDD | 2022 MaxDD | 2023-24 return | Notes |
|---|---|---|---|---|---|---|
| v19 baseline | 673.3% | 1.01 | 46.8% | 31.1% | 43.6% | High return, high DD |
| mode_v1 | 343.6% | 1.03 | **42.3%** | **16.5%** | 43.0% | Best DD control in family |
| **learned_mode_v1 (trained)** | **507.0%** | **0.87** | **50.0%** | **22.6%** | **63.8%** | Trained actor; higher return than mode_v1 but DD worse than v19 |
| v19 + learned router (fixed) | 336.7% | 0.85 | 38.2% | 18.2% | 51.0% | Fixed actor; smarter than rule router |
| v19 + rt0.6 budget (fixed) | 669.5% | 1.04 | 42.1% | 29.4% | 43.3% | **Best mechanical overlay** — return unchanged, DD down 4.7pts |
| v19 + learned + budget0.5 | 343.4% | 0.87 | 35.8% | 18.2% | 49.6% | Hybrid on fixed actor |
| actor ensemble (switching) | ~40% | ~0.85 | ~10% | ~13% | ~5% | Complete failure — do not pursue |

## Key Findings

### 1. The 30% MaxDD target is extremely aggressive for 3x leveraged ETF

- mode_v1 already achieves **42.3% proxy MaxDD** — this is likely close to the practical frontier for a single SOXL strategy without external hedging.
- To reach 30%, the strategy would need to avoid almost all major drawdown episodes in the 2020-2026 window. Given SOXL's 3x leverage, this requires either:
  - Perfect market timing (impossible)
  - Permanent de-risking that sacrifices most bull capture
  - External hedging instruments (options, shorts) not available in current harness

### 2. Learned mode classifier works, but training adapts away from its benefits

- On **fixed v19 actor**, learned router achieves 38.2% MaxDD vs rule router's 42.3% (mode_v1).
- But when actor is **trained** under learned router, it learns to exploit the router's permissiveness, pushing MaxDD back to 50.0%.
- The training reward function does not strongly enough penalize the path-level drawdowns that drive proxy MaxDD.

### 3. Regime-conditional budget masking is the best overlay

- **rt0.6_b0.3 on v19**: return 669.5% (vs 673.3% baseline), Sharpe 1.04, MaxDD 42.1%.
- This is a **genuine improvement** with virtually no return cost.
- It works because it only constrains during genuine high-risk periods (regime_risk >= 0.6).
- It cannot push MaxDD below ~40% without meaningful return sacrifice.

### 4. Actor ensemble (v19 + mode_v1 switching) fails completely

- Direct switching between two actors with different risk profiles creates incoherent behavior.
- Return collapses to ~40% because the aggressive actor is trapped by the defensive actor's constraints during transitions.

### 5. The real problem is transition discrimination

- experiment-landscape-review already identified this as the #1 research gap.
- All approaches tested here (budget, learned router, ensemble) attack the problem from different angles but share the same limitation: **no single mechanism can simultaneously capture bull trends and avoid bear drawdowns at the 30% level for 3x leverage**.

## Decision

1. **Drop actor ensemble approach** — not viable.
2. **Keep regime-conditional budget masking as a production overlay** — it provides ~5pt MaxDD improvement at zero return cost on v19. This is the only approach with a clean positive expected value.
3. **Keep learned mode classifier as a candidate module** — it is smarter than rule-based classification, but requires either:
   - A training algorithm that respects its constraints (hard to achieve with PPO)
   - Or using it only in fixed-actor inference with the understanding that return is capped
4. **Acknowledge the 30% MaxDD target may require a fundamentally different strategy class**:
   - Multi-asset risk diversification (not single-asset SOXL)
   - Options overlay or tail-risk hedging
   - Higher-frequency intraday execution to cut losses faster
   - These are outside the current harness scope.

## Next Actions

1. Run **regime-conditional budget masking (rt0.6_b0.3) as a fixed-actor overlay** on the latest canonical actor and evaluate walk-forward.
2. If walk-forward passes, consider promoting the budget overlay to production.
3. For the 30% MaxDD goal, investigate **multi-asset allocation** (SOXL + TQQQ + UPRO) with cross-asset risk budgeting — this is the only remaining architectural direction with a theoretical path to 30% drawdown.
4. Update experiment-landscape-review.md with these conclusions.

## Retrieval tags

- learned-mode-router
- drawdown-budget
- regime-conditional
- actor-ensemble
- fundamental-frontier
- v19
- mode_v1
- transition-discrimination
- 30-percent-drawdown
