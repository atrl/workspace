---
date: 2026-04-24
title: Category-first runtime cleanup
kind: ops
profiles:
- soxl
status: kept
tags:
- checkpoint-selection
- kept
- ops
- soxl
artifacts:
- runtime/data/
- runtime/execution/
- runtime/execution/generalization/
- runtime/factor/
- runtime/factor/analysis/
- runtime/model/
- runtime/model/diagnosis/
- runtime/model/reviews/
- runtime/network/
workpad: codex_exec_plans/active/current.md
summary: Collapsed the separate evidence surface back into runtime, removed historical
  codex_exec_plans workpads, and deleted the root multi_asset data copy so iteration
  starts from category-specific runtime paths and memory indexes.
---

# Category-first runtime cleanup

## Context

The previous refactor made the repository harder to read by adding a new top-level
`evidence/` surface while historical `codex_exec_plans/completed/` and root
`multi_asset/` still remained.

## Decision

Use fewer top-level concepts:

- `codex_exec_plans/`: active workpad only.
- `memory/index.json`: historical thread retrieval.
- `runtime/<category>/`: generated outputs grouped by `model`, `factor`, `data`, `network`, `execution`, plus `soxl` for live execution.
- `data/`: physical raw/processed/qlib/cache storage.

## Implementation

- Moved former evidence artifacts into category paths:
  - `runtime/model/reviews/`
  - `runtime/model/diagnosis/`
  - `runtime/factor/analysis/`
  - `runtime/execution/generalization/`
- Replaced `scripts/build_evidence_index.py` with `scripts/build_runtime_index.py`.
- Removed historical workpads from `codex_exec_plans/completed/`; durable history now starts from `memory/index.json`.
- Removed the root `multi_asset/` data duplicate; canonical stores are `data/qlib/` and `data/cache/training/`.
- Tightened `scripts/check_source_boundaries.py` to block top-level `evidence/`, root `multi_asset/`, and historical completed workpads.

## Validation

- `PYTHONPATH=. qlib_venv/bin/python scripts/build_runtime_index.py --check`: OK.
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_source_boundaries.py`: OK.
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_research_knowledge.py`: OK, files=7, active_budget=300.
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_experiment_cards.py`: OK, packages=64, registry_rows=110.
- `PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py --check`: OK, symbols=5, panels=10, qlib=6.
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py`: OK, passed=62, failures=0.

## Next Use

For future automated iteration, pick the mechanism first, then use:

- model/checkpoint/reward outputs: `runtime/model/`
- factor outputs: `runtime/factor/`
- data refresh outputs: `runtime/data/`
- architecture outputs: `runtime/network/`
- execution/generalization outputs: `runtime/execution/`
