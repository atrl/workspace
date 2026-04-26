# Runtime

`runtime/` is the single generated-output surface. Every experiment-specific
artifact lives inside its corresponding `runtime/<direction>/cards/<name>/`
directory. Cross-cutting or generic outputs that do not belong to a single
card stay in the category root.

## Read Order

1. `runtime/index.json` for generated artifact pointers.
2. `runtime/<direction>/cards/<name>/` for experiment-specific profiles,
   notes, backtest summaries, reviews, diagnostics, walk-forward results, and
   daily CSVs.
3. `runtime/<direction>/knowledge.md` for continuity and durable conclusions.

## Category Roots (cross-cutting / generic)

| Path | Purpose |
|------|---------|
| `runtime/data/universes/` | ETF universe definitions (e.g. `3x_etf_core.yaml`). |
| `runtime/factor/` | Cross-factor IC results, correlation matrices, and rulebase hypotheses. Per-factor backtests live in `runtime/factor/cards/<factor_name>/`. |
| `runtime/model/` | Cross-experiment diagnosis documents, cross-family review syntheses, and walk-forward meta-results. Per-model outputs live in `runtime/model/cards/<name>/`. |
| `runtime/multi_asset/` | Cross-ETF comparison charts. Per-ETF detailed charts live in `runtime/multi_asset/cards/<name>/`. |
| `runtime/execution/` | Shared execution configs (`*_variants.json`) and `knowledge.md`. Per-overlay evidence lives in `runtime/execution/cards/<name>/`. |
| `runtime/soxl/paper_trading/` | Live / paper-trading execution traces. |

## Cards Discipline

- A card directory (`runtime/<direction>/cards/<name>/`) is the single home for
  everything produced by that experiment: `profile.yaml`, `card.md`, backtest
  summaries, review markdown, daily CSVs, walk-forward JSONs, and worker logs.
- Do not scatter a single experiment's outputs across multiple category roots.
- Do not create per-function subdirectories (e.g. `reviews/`, `diagnosis/`,
  `analysis/`) inside a category; either put files into the matching card or
  leave them in the category root if they are cross-cutting.

## Boundary

- Experiment intent belongs in `experiments/`, not `runtime/`.
- Durable conclusions belong in `runtime/<direction>/knowledge.md` or `docs/research_knowledge/`.
- Do not recreate a top-level `evidence/` directory.
