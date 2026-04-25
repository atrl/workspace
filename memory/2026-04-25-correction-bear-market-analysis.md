# Correction: Bear-Market Analysis Was Wrong

Date: 2026-04-25

## Error

Previous analysis concluded that v19, v22, and balanced "failed" walk-forward validation on 2022, calling it a "family-level bear-market failure." **This was incorrect.**

## Missing Context

**SOXL buy-and-hold in 2022: -86.6%**

The models "only" lost 21-32%, outperforming by 55-65 percentage points. This is successful bear-market defense, not failure.

## Corrected Assessment

| Profile | 2022 Return | vs SOXL B&H | Assessment |
|---------|------------|-------------|------------|
| balanced | -25.8% | +60.8pp | ✅ Successful defense |
| v19 | -25.2% | +61.4pp | ✅ Successful defense |
| v22 | -21.0% | +65.6pp | ✅ Successful defense |
| v19_small | -31.6% | +55.0pp | ✅ Defense OK, but gate fails on MaxDD |

## Why the Gate Rejected Them

The walk-forward promotion gate uses absolute thresholds:
- test_proxy_sharpe > 0.75
- test_max_dd < 35%

In a year where SOXL drops 86%, any strategy holding the asset will have negative returns and negative Sharpe. The gate should use **benchmark-relative metrics** (information ratio vs SOXL) instead of absolute Sharpe.

## Real Issues (Not Bear-Market Defense)

1. **Proxy MaxDD 51-55%** — High drawdown over full backtest period
2. **Recovery underperformance** — 2023-2024: models +60-79% vs SOXL B&H +182%
3. **Slow re-entry** — Models are too defensive after bear markets, miss upside

## Registry Corrections

- v19: restored to `candidate` (was incorrectly marked `dropped`)
- v22: restored to `candidate` (was incorrectly marked `dropped`)
- v19_small: remains `dropped` (fixed-split test_2022 MaxDD 38.3% exceeds 35% limit)
- v19_tiny: remains `dropped` (baseline v19_small failed)

## Files Updated

- `runtime/model/reviews/v19_walk_forward_review.md` — corrected
- `docs/experiment-landscape-review.md` — corrected
- `configs/soxl/profile_registry.yaml` — corrected
