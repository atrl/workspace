# Runtime

`runtime/` is the single generated-output surface. It is organized by iteration
category so future agents can retrieve outputs without guessing which top-level
directory owns them.

## Read Order

1. `runtime/index.json` for generated artifact pointers.
2. `runtime/model/` for model reviews, checkpoint diagnostics, policy comparisons, and post-train outputs.
3. `runtime/factor/` for factor scans, feature importance, and IC analysis.
4. `runtime/data/` for dataset refresh and data-quality evidence.
5. `runtime/network/` for architecture-specific diagnostics.
6. `runtime/execution/` for dashboards, execution overlays, and cross-ETF transfer evidence.
7. `runtime/soxl/` for paper-trading and live execution traces.

## Boundary

- Experiment intent belongs in `experiments/`, not `runtime/`.
- Durable conclusions belong in `memory/` or `docs/research_knowledge/`.
- Do not recreate a top-level `evidence/` directory.
