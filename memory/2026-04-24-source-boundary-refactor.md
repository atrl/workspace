---
date: 2026-04-24
title: Source boundary refactor for evidence, runtime, experiments, memory, and data
kind: ops
profiles:
- soxl
status: kept
tags:
- kept
- ops
- soxl
artifacts:
- runtime/execution/generalization
- runtime/factor/analysis
- runtime/index.json
- runtime/model/diagnosis
- runtime/model/reviews
- runtime/soxl/
workpad: codex_exec_plans/active/current.md
summary: Runtime is now execution state only, generated research artifacts live under
  evidence, 3x ETF universe config is explicit, and boundary drift is checked by the
  harness.
---

# Source boundary refactor for evidence, runtime, and data

## Context

The previous cleanup still left responsibilities blurred between `memory/`,
`experiments/`, `production/`, `runtime/`, and physical data paths.

## Decision

Keep responsibilities narrow:

- `runtime/`: live execution state only, currently `runtime/soxl/`.
- `runtime/`: generated research and review artifacts, indexed by `runtime/index.json`.
- `experiments/`: runnable intent cards, not result dumps.
- `memory/`: continuity decisions and next-use notes, not raw logs.
- `configs/universes/`: ETF universe definitions independent from physical data layout.

## Implementation

- Moved research outputs from `runtime/post_train_review`, `runtime/generalization`, `runtime/diagnosis`, and `runtime/feature_analysis` into `runtime/`.
- Added `scripts/build_runtime_index.py` for generated evidence retrieval.
- Added `scripts/check_source_boundaries.py` and wired it into `scripts/check_harness.py`.
- Added `configs/universes/3x_etf_core.yaml` and `src/soxl_quant/core/universe.py`.
- Removed active `data/multi_asset` coupling from multi-asset source bootstrapping; data stores now resolve through `data/qlib`, `data/processed`, and `data/cache`.

## Validation

- `PYTHONPATH=. qlib_venv/bin/python scripts/check_research_knowledge.py`: OK, files=7, active_budget=300.
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_experiment_cards.py`: OK, packages=64, registry_rows=110.
- `PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py --check`: OK, symbols=5, panels=10, qlib=6.
- `PYTHONPATH=. qlib_venv/bin/python scripts/build_runtime_index.py --check`: OK.
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_source_boundaries.py`: OK.
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py`: OK, passed=62, failures=0.

## Next Use

Before future automated iteration, start from `codex_exec_plans/active/current.md`, then use `docs/research_knowledge/index.md`, `experiments/index.md`, `runtime/index.json`, and `data/datasets.yaml`. Put generated outputs under the matching `runtime/<category>/` path.
