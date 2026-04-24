---
date: '2026-04-21'
title: 2026-04-21 experiment landscape review and research gate hardening
kind: decision
profiles:
- balanced
- balanced_rule_residual
- balanced_rule_residual_v1
- balanced_rule_residual_v19
- balanced_rule_residual_v22
status: kept
tags:
- balanced
- balanced_rule_residual
- balanced_rule_residual_v1
- balanced_rule_residual_v19
- balanced_rule_residual_v22
- decision
- kept
- stress-regime
- transition-features
artifacts: []
workpad: memory/index.json
summary: Added a durable repo-wide experiment landscape review and hardened `AGENTS.md`
  so new experiments now require an explicit baseline, closest success/failure case,
  external evidence class, experiment card, and comparison-table row. The repo should
  now bias toward narrower, mechanism-new branches instead of repeating reward, threshold,
  or bundled-structure loops.
---

# 2026-04-21 experiment landscape review and research gate hardening

## Context

The repo had enough historical evidence to explain current strengths and repeated failures, but that knowledge was scattered across `memory/`, experiment notes, and pairwise post-train reviews.

## What Changed

- Added `docs/experiment-landscape-review.md` as the durable comparison surface for historical major branches.
- Upgraded `AGENTS.md` with hard pre-run requirements:
  - explicit experiment entry gate
  - external-evidence requirement
  - anti-repetition rule
  - experiment card contract
  - mandatory comparison-table update
  - pre-run reflection on strong vs weak windows
- Updated the active workpad so the new document becomes the default pre-run briefing artifact.

## Evidence

- Mainline conclusion remains stable:
  - canonical baseline: `balanced`
  - strongest family: `balanced_rule_residual`
- Best clean architecture gain:
  - `balanced_rule_residual_v1`
- Best high-return GRU point:
  - `balanced_rule_residual_v19`
- Best external-risk feature probe:
  - `balanced_rule_residual_v22`
- Most repeated low-value loops:
  - reward-only coefficient tuning
  - wide transition-feature batches
  - bundled structural changes
  - aggressive floor/cap branches that re-break the stress gate

## Decision

- Keep the new landscape review as the durable historical summary.
- Keep pairwise reviews for local comparison, but do not rely on them as the only record.
- Treat future experiment proposals without a new mechanism or explicit historical contrast as out of policy.

## Next Use

Before the next experiment:

1. read `docs/experiment-landscape-review.md`
2. name the closest success case
3. name the closest failure case
4. state the new mechanism and the row to be added back into the comparison table
