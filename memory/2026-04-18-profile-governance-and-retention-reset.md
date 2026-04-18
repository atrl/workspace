---
date: '2026-04-18'
title: 2026-04-18 Profile Governance And Retention Reset
kind: review
profiles:
- soxl
status: dropped
tags:
- checkpoint-selection
- dropped
- eval-2022
- review
- soxl
- walk-forward
artifacts:
- artifacts/models/soxl_balanced_actor.pt
- artifacts/models/soxl_balanced_checkpoint_review_actor.pt
- artifacts/models/soxl_balanced_wf_2020_actor.pt
- artifacts/models/soxl_balanced_wf_2022_actor.pt
- artifacts/models/soxl_balanced_wf_2023_2024_actor.pt
- artifacts/models/soxl_smoke_actor.pt
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: 'The harness had reached the point where `rl.py`, `cli.py`, and the profile
  set were too entangled to answer basic questions reliably: which profile is canonical
  which branches are just ablations which actor checkpoints should still exist wh'
---

# 2026-04-18 Profile Governance And Retention Reset

## Context

The harness had reached the point where `rl.py`, `cli.py`, and the profile set
were too entangled to answer basic questions reliably:

- which profile is canonical
- which branches are just ablations
- which actor checkpoints should still exist
- whether a behavior change came from reward, execution, observation, or cap logic

This loop tightened the repo around explicit ownership and retention rules.

## Evidence

### Structural split

- Added `src/soxl_quant/config.py` for typed config dataclasses and profile
  loading.
- Added `src/soxl_quant/registry.py` and
  `configs/soxl/profile_registry.yaml` so profile lifecycle is repo-owned.
- Narrowed `src/soxl_quant/settings.py` to paths, constants, and artifact naming
  helpers.
- Made `src/soxl_quant/cli.py` discover profile names dynamically from config
  files instead of a hand-maintained `choices=[...]` list.
- Split `src/soxl_quant/rl.py` by extracting:
  - `src/soxl_quant/rl_execution.py`
  - `src/soxl_quant/rl_positioning.py`
  - `src/soxl_quant/rl_observation.py`

### Operating contract

- Updated `AGENTS.md` to add:
  - profile lifecycle and baseline rules
  - code-boundary rules for `settings.py` / `config.py` / `cli.py`
  - registry-driven retention and cleanup rules

### Cleanup

- Reworked `scripts/prune_artifacts.py` to use
  `configs/soxl/profile_registry.yaml` instead of hard-coded keep/delete lists.
- Two-pass cleanup result:
  - initial cleanup freed `3,971,622` bytes
  - tightened actor-tag rules freed another `498,421` bytes
- Final kept actor inventory:
  - `artifacts/models/soxl_balanced_actor.pt`
  - `artifacts/models/soxl_balanced_checkpoint_review_actor.pt`
  - `artifacts/models/soxl_balanced_wf_2020_actor.pt`
  - `artifacts/models/soxl_balanced_wf_2022_actor.pt`
  - `artifacts/models/soxl_balanced_wf_2023_2024_actor.pt`
  - `artifacts/models/soxl_smoke_actor.pt`

### Validation

- `qlib_venv/bin/python -m py_compile ...` passed for the new split modules.
- `PYTHONPATH=. qlib_venv/bin/python tests/test_artifact_paths.py`
- `PYTHONPATH=. qlib_venv/bin/python tests/test_position_cap.py`
- `PYTHONPATH=. qlib_venv/bin/python -c "import tests.test_profile_registry ..."`
- `PYTHONPATH=. qlib_venv/bin/python -c "import tests.test_structural_model ..."`

All targeted checks passed.

## Decision

- Keep the profile registry as the source of truth for canonical/candidate/
  dropped/archived state.
- Keep summary/review evidence broadly, but only keep registry-approved actor
  bundles.
- Continue the code split next from `rl.py` into reward / policy / evaluation
  modules instead of adding more logic back into the facade.
