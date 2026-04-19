---
date: 2026-04-19
title: AGENTS Contract Refresh
kind: other
profiles: []
status: reference
tags:
- completed
- refactor
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: Refreshed AGENTS.md to match the current app/policy/core split, declared
  cli.py and rl.py compatibility-only, added script-boundary rules, and documented
  preferred read order for experiments, runtime, and memory.
---

# AGENTS Contract Refresh

## Context

The codebase has already moved away from monolithic `cli.py` and `rl.py`, but the repo contract still described the older ownership model too loosely.

## What Changed

- documented `src/soxl_quant/app/` module ownership in detail
- marked top-level `cli.py` and `rl.py` as compatibility facades only
- added rules for repo-owned scripts to import `app/`, `policy/`, `core/`, or `runtime.py` directly
- added read-order guidance for `experiments/`, `runtime/`, and `memory/`
- added a cleanup note for `__pycache__/` and stale generated files

## Decision

Future refactors should update `AGENTS.md` as part of the same loop whenever ownership boundaries change, so the contract stays ahead of drift rather than documenting it later.
