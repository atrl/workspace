---
date: 2026-04-19
title: CLI Shell Shrink
kind: other
profiles: []
status: reference
tags:
- other
- reference
artifacts: []
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: Split runtime bootstrap and end-of-run reporting out of app/cli.py so the
  CLI entrypoint is a thin orchestration shell rather than a mixed setup/execution/reporting
  file.
---

# CLI Shell Shrink

## Context

After splitting training, workers, sources, and selection, `app/cli.py` still mixed setup, dataset bootstrap, profile loop, and final reporting.

## What Changed

- added `src/soxl_quant/app/bootstrap.py`
- added `src/soxl_quant/app/reporting.py`
- reduced `src/soxl_quant/app/cli.py` to argument parsing and high-level orchestration

## Evidence

- `PYTHONPATH=. qlib_venv/bin/python -m py_compile src/soxl_quant/app/bootstrap.py src/soxl_quant/app/reporting.py src/soxl_quant/app/cli.py src/soxl_quant/app/train.py src/soxl_quant/cli.py`: passed
- `PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py`: `passed=32 failures=0`

## Decision

Keep bootstrap and reporting logic outside the CLI shell so later CLI changes stay local to command orchestration.
