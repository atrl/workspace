# Current SOXL Harness Workpad

Status: active
Owner: Codex
Last updated: 2026-04-24 CST

## Goal

Restructure the harness knowledge, workpad, experiment, skill, and data surfaces so future agents can start from compact indexes and then drill into category-specific evidence.

## Current Baselines

- Canonical strategy baseline: `balanced` via `experiments/canonical/balanced/`.
- Retained frontier family: `balanced_rule_residual`, with recent retained evidence summarized in `docs/research_knowledge/model_selection.md` and `docs/research_knowledge/execution_overlays.md`.
- Multi-asset caution: prior shared-actor 3x ETF training failed without generic reward, volatility normalization, or asset-specific heads; see `docs/research_knowledge/multi_asset.md`.

## Active Task

- Direction: `harness` cleanup for task routing and knowledge retrieval.
- Implementation: make every future iteration choose a primary task direction before reading memory or runtime artifacts.
- Acceptance criteria:
  - active workpads stay short and indexed: done via `current.md` plus `memory/index.json`
  - research knowledge is split by mechanism category: done via `docs/research_knowledge/`
  - task direction routes connect knowledge -> skill -> memory query -> runtime path: done via `docs/research_knowledge/task_router.json`
  - experiment packages are concise cards, not runtime logs: done via card checks
  - data paths are registry-backed and ready for automated refresh checks: done via `data/datasets.yaml`
  - generated research artifacts are category-organized under `runtime/`
  - stale boundary regressions are mechanically blocked: done via `scripts/check_source_boundaries.py` and `scripts/check_research_task_router.py`

## Evidence Links

- Historical retrieval: `memory/index.json`
- Knowledge index: `docs/research_knowledge/index.md`
- Experiment index: `experiments/index.md`
- Data registry: `data/datasets.yaml`
- Runtime index: `runtime/index.json`
- Task router: `docs/research_knowledge/task_router.json`

## Commands

- `PYTHONPATH=. qlib_venv/bin/python scripts/check_research_knowledge.py`
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_experiment_cards.py`
- `PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py --check`
- `PYTHONPATH=. qlib_venv/bin/python scripts/build_runtime_index.py --check`
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_source_boundaries.py`
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py`

## Latest Evidence

- `scripts/check_research_knowledge.py`: OK, files=7, active_budget=300.
- `scripts/check_experiment_cards.py`: OK, packages=64, registry_rows=110.
- `scripts/update_data_registry.py --check`: OK, symbols=5, panels=10, qlib=6.
- `scripts/build_runtime_index.py --check`: OK.
- `scripts/check_source_boundaries.py`: OK.
- `scripts/check_research_task_router.py`: OK, directions=7.
- `scripts/check_harness.py`: OK, passed=62, failures=0, py_compile targets=30.

## Next Action

Validate, commit, and push the task-direction router cleanup.
