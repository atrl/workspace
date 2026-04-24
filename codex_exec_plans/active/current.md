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

- Implementation: strong migration of source-of-truth surfaces.
- Acceptance criteria:
  - active workpads stay short and indexed
  - research knowledge is split by mechanism category
  - factor skills are generic leveraged-ETF workflows with SOXL defaults
  - experiment packages are concise cards, not runtime logs
  - data paths are registry-backed and ready for automated refresh checks

## Evidence Links

- Historical main loop: `codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md`
- Historical feature recall: `codex_exec_plans/completed/2026-04/2026-04-17-historical-balanced-feature-recall.md`
- Knowledge index: `docs/research_knowledge/index.md`
- Experiment index: `experiments/index.md`
- Data registry: `data/datasets.yaml`
- Runtime evidence index: `runtime/evidence_index.json`

## Commands

- `PYTHONPATH=. qlib_venv/bin/python scripts/check_research_knowledge.py`
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_experiment_cards.py`
- `PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py --check`
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py`

## Next Action

Run validation after migration and record the result in `memory/` before commit/push.
