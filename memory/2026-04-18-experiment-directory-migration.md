---
date: '2026-04-18'
title: 2026-04-18 Experiment Directory Migration
kind: failure
profiles:
- balanced
- balanced_checkpoint_review
- smoke
- soxl
status: dropped
tags:
- balanced
- balanced_checkpoint_review
- checkpoint-selection
- dropped
- failure
- smoke
- soxl
artifacts: []
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: The flat `configs/soxl/*.yaml` layout still made profile intent too opaque
  even after adding a registry. The next step was to start colocating profile config
  with experiment-owned notes and expectations. Added `experiments/README.md`. Added
---

# 2026-04-18 Experiment Directory Migration

## Context

The flat `configs/soxl/*.yaml` layout still made profile intent too opaque even
after adding a registry. The next step was to start colocating profile config
with experiment-owned notes and expectations.

## Evidence

- Added `experiments/README.md`.
- Added experiment packages for the three current anchors:
  - `experiments/canonical/balanced/`
  - `experiments/candidate/balanced_checkpoint_review/`
  - `experiments/canonical/smoke/`
- Each package now keeps:
  - `profile.yaml`
  - `notes.md`
  - `expectations.md`
- `src/soxl_quant/config.py` now resolves profiles in this order:
  1. `experiments/*/<name>/profile.yaml`
  2. legacy `configs/soxl/<name>.yaml`
- Legacy flat YAMLs for those three profiles were left in place as compatibility
  paths, but now include explicit migration comments pointing back to the
  experiment-owned file.
- Registry notes for `balanced`, `balanced_checkpoint_review`, and `smoke` now
  point to experiment-local notes instead of generic workpad paths.

## Validation

- `PYTHONPATH=. qlib_venv/bin/python -m py_compile src/soxl_quant/config.py src/soxl_quant/registry.py src/soxl_quant/cli.py`
- `PYTHONPATH=. qlib_venv/bin/python -c "import tests.test_profile_registry ..."`
- `PYTHONPATH=. qlib_venv/bin/python tests/test_checkpoint_review_assets.py`
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py`

All checks passed. `scripts/check_harness.py` remained green at
`passed=32 failures=0`.

## Decision

- Keep experiment-directory-first loading.
- Keep legacy flat config files temporarily for compatibility.
- Continue migration by moving the active balanced-family candidates before
  cleaning up archived profiles.
