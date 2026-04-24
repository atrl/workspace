# Runtime Evidence

`runtime/` contains generated evidence only. Do not use it as the primary experiment card or memory surface.

## Read Order

1. `runtime/evidence_index.json` for generated evidence pointers.
2. `runtime/post_train_review/` for pairwise or local run reviews.
3. `runtime/generalization/` for cross-ETF and dashboard evidence.
4. `runtime/diagnosis/` for trade-quality, Kelly, and slice diagnostics.
5. `runtime/soxl/` for paper-trading and execution traces.

## Durable Summaries

- Major historical conclusions belong in `docs/experiment-landscape-review.md` and `docs/research_knowledge/`.
- Run-specific decisions belong in `memory/`.
- Experiment intent belongs in `experiments/`.
