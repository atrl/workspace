---
date: '2026-04-21'
title: 2026-04-21 historical integration policy
kind: decision
profiles:
- balanced
- balanced_rule_residual
- balanced_rule_residual_v19
- balanced_rule_residual_v22
- smoke_rule_residual_v1
status: active
tags:
- active
- balanced
- balanced_rule_residual
- balanced_rule_residual_v19
- balanced_rule_residual_v22
- checkpoint-selection
- decision
- execution-overlay
- smoke_rule_residual_v1
- transition-features
artifacts: []
workpad: memory/index.json
summary: Added a durable historical integration policy to `docs/experiment-landscape-review.md`.
  The repo should now retain a single core stack (`balanced`, `balanced_rule_residual`,
  `v19`, and the harness governance path), treat `v22`, checkpoint review, smoke residual
  checks, and overlay studies as optional modules, and permanently demote reward-only
  tuning, broad transition bundles, narrow transition retrains without selective execution,
  generic overlay retuning, and bundled structural retries to evidence-only directions.
---

# 2026-04-21 historical integration policy

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Purpose

Future work should stop "integrating" history by stacking together every branch
that once looked somewhat useful. Instead, the repo now keeps:

- a stable core stack
- a small optional module set
- a permanent evidence-only discard set

## Core keep set

- `balanced`
- `balanced_rule_residual`
- `balanced_rule_residual_v19`
- harness governance surfaces:
  - registry
  - archive
  - memory
  - comparison table
- fixed-actor execution overlay evaluation path

## Optional module set

- `balanced_rule_residual_v22`
- checkpoint review / early selection
- `smoke_rule_residual_v1`
- existing overlay routing studies as evaluation tools

## Permanent drop set

- reward-only tuning
- broad transition feature bundles
- narrow transition retrains without selective execution
- generic overlay retuning with no new execution state
- bundled structural retries

## Operational rule

Every new experiment should start from the retained core and add at most one
new optional module or one new mechanism. If a proposal only changes
thresholds, penalties, or naming inside a dropped direction, it should default
to not running.

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
