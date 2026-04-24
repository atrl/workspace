---
date: '2026-04-21'
title: 2026-04-21 memory and experiments cleanup
kind: ops
profiles:
- balanced_rule_residual_v19
- balanced_rule_residual_v22
- smoke_rule_residual_v1
status: kept
tags:
- balanced_rule_residual_v19
- balanced_rule_residual_v22
- kept
- ops
- smoke_rule_residual_v1
artifacts: []
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: Reduced `experiments/candidate/` to the real active set (`balanced_rule_residual_v19`,
  `balanced_rule_residual_v22`, `smoke_rule_residual_v1`), moved older packages into
  `experiments/archive/`, clarified the split between experiment cards and memory
  result notes, and shortened duplicated active notes so result narratives now live
  in `memory/`.
---

# 2026-04-21 memory and experiments cleanup

## Context

`experiments/` had accumulated too many historical packages under `candidate/`, and several package notes were duplicating result narratives already captured in `memory/`.

## What Changed

- Kept only the real active set in `experiments/candidate/`:
  - `balanced_rule_residual_v19`
  - `balanced_rule_residual_v22`
  - `smoke_rule_residual_v1`
- Moved older historical packages to `experiments/archive/`.
- Updated `profile_registry.yaml` notes references to point at the new archive paths.
- Rewrote the active `v19` / `v22` package notes to be short experiment cards with evidence links instead of full duplicated result writeups.
- Clarified scope in:
  - `experiments/README.md`
  - `experiments/archive/README.md`
  - `memory/README.md`

## Decision

- `experiments/` is now the home of active cards plus frozen archive packages.
- `memory/` is the durable result and decision layer.
- If a run already has a proper `memory` note, do not repeat the full result narrative in the experiment package.

## Next Use

When opening old work:

1. start from `docs/experiment-landscape-review.md`
2. check `profile_registry.yaml`
3. open the active or archived experiment package
4. then open the linked memory note for the actual result
