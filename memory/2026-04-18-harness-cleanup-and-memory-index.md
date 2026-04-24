---
date: '2026-04-18'
title: '2026-04-18: harness cleanup, indexed memory retrieval, and soft artifact prune'
kind: ops
profiles:
- balanced
- balanced_floor40_sym
- balanced_regime_gate
status: kept
tags:
- balanced
- balanced_floor40_sym
- balanced_regime_gate
- checkpoint-selection
- dashboard
- eval-2022
- kept
- ops
- transition-features
artifacts:
- runtime/execution/generalization/dashboard.html
- runtime/source
workpad: memory/index.json
summary: Replaced the broken pytest expectation with a repo-owned check_harness entrypoint,
  normalized memory into front-matter plus index.json, deleted low-value scripts and
  superseded actor checkpoints, and upgraded the static dashboard with evidence navigation
  and persisted state.
---

# 2026-04-18: harness cleanup, indexed memory retrieval, and soft artifact prune

## Context

The repo had accumulated three classes of avoidable friction:

- docs still implied `pytest` was the default check path even though `qlib_venv` did not ship it
- `memory/` had drifted into mixed formats that encouraged full-directory rereads
- `artifacts/models/` and `scripts/` contained stale outputs that made future loops noisier than necessary

## What Changed

- Added `scripts/check_harness.py` as the repo-owned validation entrypoint.
- Added `scripts/memory_tools.py`, `scripts/normalize_memory.py`, and `scripts/memory_search.py`.
- Normalized existing `memory/*.md` files into a shared YAML-front-matter schema and generated `memory/index.json`.
- Added `scripts/prune_artifacts.py` and generated `artifacts/retention_manifest.json`.
- Deleted six low-reference one-off scripts from `scripts/`.
- Rebuilt the static dashboard via `scripts/build_backtest_dashboard.py` with:
  - decision snapshot badges
  - runtime/source links
  - local state persistence for ETF/chart/range selections
  - clearer delta rows in comparison tables

## Evidence

- Validation:
  - `qlib_venv/bin/python scripts/check_harness.py`
  - result: `21` assertion checks passed, `13` `py_compile` targets passed
- Memory retrieval:
  - `qlib_venv/bin/python scripts/memory_search.py balanced 2022 exposure --status dropped --top-k 5`
  - result: surfaced recent `balanced_regime_gate*`, `balanced_floor40_sym`, and transition-feature failures first
- Artifact prune:
  - `qlib_venv/bin/python scripts/prune_artifacts.py --apply`
  - result: deleted `15` superseded actor checkpoints, freed `811,339` bytes
- Dashboard rebuild:
  - `qlib_venv/bin/python scripts/build_backtest_dashboard.py`
  - result: refreshed `runtime/execution/generalization/dashboard.html`

## Decision

- Keep `memory/index.json` plus `scripts/memory_search.py` as the default retrieval path before new training.
- Keep summary JSON and review outputs as evidence; preferentially delete redundant actor checkpoints first.
- Keep the dashboard as a static generated page for now.
- Drop the old `pytest`-first assumption for this repo.

## Next Use

Before starting another training or replay loop:

1. `qlib_venv/bin/python scripts/memory_search.py balanced --status dropped --top-k 5`
2. `qlib_venv/bin/python scripts/check_harness.py`
3. review `artifacts/retention_manifest.json` if artifact churn is expected
