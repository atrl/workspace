# Research Knowledge Index

This directory is the mechanism-level recall surface for SOXL and related leveraged-ETF strategy work. Start here before opening runtime logs or long memory notes.

## Categories

| Category | File | Use when |
|---|---|---|
| Factors | `factors.md` | evaluating feature hypotheses, rulebase-derived signals, IC, or signal validation |
| Data | `data.md` | refreshing CSVs, qlib stores, feature caches, or external volatility data |
| Model selection | `model_selection.md` | choosing checkpoints, promotion gates, fixed splits, or early stopping |
| Networks | `networks.md` | changing GRU/attention/MoE/asset-head architecture |
| Rewards | `rewards.md` | changing reward penalties, opportunity cost, Kelly, imitation, or activity terms |
| Execution overlays | `execution_overlays.md` | changing fixed-actor risk budgets, mode routing, position caps, or runtime mapping |
| Multi-asset | `multi_asset.md` | testing SOXL logic against TQQQ/UPRO/SPXL/TECL or shared 3x ETF training |

## Retrieval Rule

1. Read the category file for the proposed mechanism.
2. Check `docs/experiment-landscape-review.md` for the master comparison table.
3. Query `memory/index.json` through `scripts/memory_search.py` for exact evidence.
4. Open runtime artifacts only after a category file or memory note points to them.

## Canonical Anchors

- `balanced` remains the control baseline.
- `balanced_rule_residual` is the successful residual family.
- `configs/soxl/profile_registry.yaml` remains the profile lifecycle source of truth.
- `runtime/` remains generated evidence; do not move runtime logs into experiment cards.
