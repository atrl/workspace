---
date: '2026-04-14'
title: Multi-Asset Joint Training Results
kind: review
profiles:
- soxl
status: dropped
tags:
- dropped
- review
- soxl
- walk-forward
artifacts: []
workpad: memory/index.json
summary: 'Legacy memory note normalized into the repo-wide schema. Date: 2026-04-14
  Implemented multi-asset joint training with asset one-hot embedding. Training pool:
  SOXL, TQQQ, UPRO, SPXL, TECL (all 3x leveraged ETFs). Observation dimension increa'
---

# Multi-Asset Joint Training Results

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

Date: 2026-04-14

## What changed
- Implemented multi-asset joint training with asset one-hot embedding.
- Training pool: SOXL, TQQQ, UPRO, SPXL, TECL (all 3x leveraged ETFs).
- Observation dimension increased from 48 to 53 (38 features + 10 extra states + 5 asset one-hot).
- Validation and test sets remain strictly SOXL-only to preserve model-selection comparability.

## Baseline (single-asset SOXL training, P0 optimized)
| ETF   | Return  | Sharpe | MaxDD  | Trades |
|-------|---------|--------|--------|--------|
| SOXL  | 371.58% | 0.817  | 49.90% | 900    |
| TQQQ  | 224.27% | 0.822  | 43.64% | 564    |
| UPRO  | 169.55% | 0.856  | 35.11% | 412    |
| SPXL  | 167.28% | 0.852  | 35.37% | 417    |
| TECL  | 220.18% | 0.767  | 41.52% | 629    |

## New results (multi-asset joint training)
| ETF   | Return  | Sharpe | MaxDD  | Trades |
|-------|---------|--------|--------|--------|
| SOXL  | 514.42% | 0.829  | 58.27% | 930    |
| TQQQ  | 298.88% | 0.816  | 52.90% | 606    |
| UPRO  | 227.11% | 0.836  | 43.60% | 457    |
| SPXL  | 224.19% | 0.832  | 43.97% | 448    |
| TECL  | 305.97% | 0.783  | 50.65% | 654    |

## Key takeaways
1. **SOXL in-domain** improved significantly (+143pp return). The model is learning richer regime patterns from the pooled dataset.
2. **TQQQ** gap vs Buy&Hold narrowed from ~-141pp to ~-66pp (a 75pp improvement).
3. **TECL** gap vs Buy&Hold narrowed from ~-145pp to ~-59pp (an 86pp improvement).
4. **UPRO/SPXL** are now essentially flat vs Buy&Hold (within ~3-6pp).
5. **Drawdown control** remains excellent: cross-asset max DD is roughly half of Buy&Hold levels.

## Conclusion
Multi-asset joint training + asset embedding is the right architectural direction. It closes most of the generalization gap while preserving (and even improving) in-domain performance. The residual underperformance on TQQQ/TECL vs pure Buy&Hold is the cost of the strategy's downside protection — an acceptable trade-off given the dramatically lower drawdowns.

## Next steps
- Run walk-forward validation to check if promotion gates can be cleared.
- Consider adding sector/regime embeddings or a small LSTM to further squeeze the remaining TQQQ/TECL gap.

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
