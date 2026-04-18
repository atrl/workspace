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
summary: '`memory/` is the continuity layer for decisions, failures, reviews, and
  reusable session context that should survive chat history. Every `memory/*.md` file
  uses YAML front matter plus Markdown body sections. Required front-matter fields:
  `d'
---

# Memory

`memory/` is the continuity layer for decisions, failures, reviews, and reusable session context that should survive chat history.

## Canonical Format

Every `memory/*.md` file uses YAML front matter plus Markdown body sections.

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

## Retrieval Workflow

Do not bulk-read every memory file before a new loop.

Use the generated index first:

- `qlib_venv/bin/python scripts/memory_search.py balanced --status dropped --top-k 5`
- `qlib_venv/bin/python scripts/memory_search.py 2022 exposure --profile balanced`
- `qlib_venv/bin/python scripts/memory_search.py overlay --tag execution-overlay`

Open only the top-matching files unless the first pass is insufficient.

## Write Rules

- Put live task execution state in `codex_exec_plans/active/`, not in memory.
- Put durable keep/drop decisions and recurring failure patterns in memory.
- When a memory note references a specific artifact or review file, include it in the `artifacts` field so future retrieval and pruning stay accurate.
