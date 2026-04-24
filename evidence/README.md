# Evidence

`evidence/` contains generated research outputs and review artifacts. It is the
retrieval layer for backtest reviews, diagnostics, dashboards, and cross-ETF
generalization evidence.

## Read Order

1. `index.json` for generated evidence pointers.
2. `post_train_review/` for pairwise or local run reviews.
3. `generalization/` for cross-ETF and dashboard evidence.
4. `diagnosis/` for trade-quality, Kelly, and slice diagnostics.
5. `feature_analysis/` for feature scans and attribution outputs.

## Boundary

- Do not store live execution state here; use `runtime/soxl/`.
- Do not store experiment intent here; use `experiments/`.
- Do not store durable conclusions here; summarize them in `memory/` or
  `docs/research_knowledge/`.
