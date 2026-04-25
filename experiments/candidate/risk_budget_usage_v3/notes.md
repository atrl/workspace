---
title: risk_budget_usage_v3
direction: model
status: candidate
---

# risk_budget_usage_v3

## Motivation

v1 proved budget-usage architecture works. v2 added mode-conditional reward
shaping. v3 tests: **let Actor directly observe current mode** (one-hot) to
learn mode-specific usage policies end-to-end.

Hypothesis: explicit mode input reduces sample complexity vs inferring mode
indirectly from raw features.

## External Reference

- Mode-conditioned policies in hierarchical RL (Sutton et al. 1999)
- Task-conditional policy networks (Hausman et al. 2018)

## Configuration Delta vs v2

```yaml
model:
  use_mode_conditioned_input: true  # append 4-dim mode one-hot to obs
```

## Result Summary

| Slice | Return | Sharpe | MaxDD | Trades |
|-------|--------|--------|-------|--------|
| test (full) | 434.1% | 0.87 | 47.5% | 295 |
| wf_2020 | 19.3% | 0.29 | 19.5% | 36 |
| wf_2022 | 59.9% | 0.39 | **59.0%** | 32 |
| wf_2023_2024 | 13.0% | 0.22 | 23.4% | 40 |

> Proxy eval: Sharpe 1.074, MaxDD 21.4%. Significant proxy-to-backtest gap.

**❌ PROMOTION BLOCKED.** Walk-forward generalization failed.

## Decision

Mode-conditioned actor input improves proxy metrics but does NOT solve
generalization. Model overfit to the fixed test period.

## Evidence Links

- Detailed review: `runtime/model/reviews/risk_budget_usage_v3_wf_review.md`
- Memory record: `memory/2026-04-24-risk-budget-v3-wf-failure.md`
- Landscape table: `docs/experiment-landscape-review.md`
