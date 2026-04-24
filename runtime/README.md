# Runtime State

`runtime/` contains execution state only. Do not use it for research evidence,
experiment cards, or continuity notes.

## Read Order

1. `runtime/soxl/` for paper-trading and execution traces.
2. `production/` for broker adapters and operational scripts.
3. `evidence/index.json` for generated research and review outputs.

## Boundary

- Research outputs belong in `evidence/`.
- Experiment intent belongs in `experiments/`.
- Run-specific decisions and continuity belong in `memory/`.
