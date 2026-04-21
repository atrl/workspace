---
date: ''
title: Memory
kind: failure
profiles:
- balanced
status: dropped
tags:
- balanced
- dropped
- eval-2022
- execution-overlay
- failure
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: '`memory/` is the continuity layer for durable decisions, failures, reviews,
  and reusable session context that should survive chat history. Do not scan all notes
  by hand first. Read in this order: 1. `memory/index.json` 2. `qlib_venv/bin/pyt'
---

# Memory

`memory/` is the continuity layer for durable decisions, failures, reviews, and reusable session context that should survive chat history.

## Read Order

Do not scan all notes by hand first.

Read in this order:

1. `memory/index.json`
2. `qlib_venv/bin/python scripts/memory_search.py ...`
3. only the top matching notes

Examples:

- `qlib_venv/bin/python scripts/memory_search.py balanced --status dropped --top-k 5`
- `qlib_venv/bin/python scripts/memory_search.py 2022 exposure --profile balanced`
- `qlib_venv/bin/python scripts/memory_search.py overlay --tag execution-overlay`

## Note Format

Every `memory/*.md` note should use YAML front matter plus a short Markdown body.

Required front-matter fields:

- `date`
- `title`
- `kind`
- `profiles`
- `status`
- `tags`
- `artifacts`
- `workpad`
- `summary`

Preferred body sections:

- `## Context`
- `## What Ran` or `## What Changed`
- `## Evidence`
- `## Decision`
- `## Next Use`

## Scope Rules

- Put live execution state in `codex_exec_plans/active/`, not in memory.
- Put durable keep/drop decisions and recurring failure patterns in memory.
- Put active experiment cards in `experiments/`, not in memory.
- Put full result narratives in `memory/` once a run has completed.
- When a note references a specific artifact or review file, include it in `artifacts` so retrieval and pruning stay aligned.
- If two notes say nearly the same thing, keep the sharper one and link the artifact path instead of repeating long narrative.

## Relationship To `experiments/`

`experiments/` and `memory/` serve different roles:

- `experiments/`
  - current or archived experiment packages
  - config + expectation card + short notes
- `memory/`
  - result
  - decision
  - failure pattern
  - continuity

If both exist for the same profile:

- `experiments/*/notes.md` should explain the intended mechanism and point to evidence
- `memory/*.md` should hold the actual outcome and keep / drop conclusion

When a package note starts duplicating the entire result writeup, shorten the
package note and keep the result in memory.
