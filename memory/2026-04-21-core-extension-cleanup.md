---
date: 2026-04-21
title: 2026-04-21 core extension cleanup
kind: failure
profiles:
- balanced_rule_residual_v22
- balanced_rule_residual_v23
status: dropped
tags:
- balanced_rule_residual_v22
- balanced_rule_residual_v23
- dropped
- execution-overlay
- failure
- transition-features
artifacts: []
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: Cleaned failed experiment mechanisms out of core runtime paths. Execution
  experiments now extend through declarative `custom_regimes`, and feature experiments
  now extend through profile-owned `feature.packs` plus optional experiment-local
  `extensions.py` files instead of env flags and shared dataclass field growth.
---

# 2026-04-21 core extension cleanup

## Context

Recent failed experiments had left too much experiment-specific logic inside
shared runtime modules:

- `ExecutionConfig` grew `release_*` fields for one dropped overlay branch
- `data.py` grew env-gated feature-schema mutation for `v22` and `v23`

That pattern makes the core path drift toward a graveyard of dropped mechanisms.

## What Changed

- Replaced release-specific execution fields with generic
  `execution.custom_regimes`
- Added profile-owned `feature.packs`
- Added experiment-local extension loading from `experiments/*/<name>/extensions.py`
- Moved archived `balanced_rule_residual_v23` transition-pack logic into its
  own experiment package
- Migrated `balanced_rule_residual_v22` to declarative `feature.packs: [vix_term]`

## Validation

- feature schema probe:
  - default `59`
  - `vix_term` pack `64`
  - `transition_v23` pack `65`
- `tests/test_runtime.py`: OK
- `tests/test_regime_features.py`: OK
- `tests/test_structural_model.py`: OK
- `scripts/check_harness.py`: `passed=39 failures=0`

## Decision

- Keep generic extension points in core:
  - `ExecutionConfig.custom_regimes`
  - `feature.packs`
  - experiment-local `extensions.py`
- Drop experiment-specific hardcoded fields and env flags from the shared path

## Rule Going Forward

If an experiment needs custom features or a new execution state:

1. put the feature pack implementation in the experiment package or a builtin
   pack registry
2. select it from profile YAML
3. do not add one-off flags or one-off fields to shared core dataclasses unless
   the mechanism is promoted into the durable baseline
