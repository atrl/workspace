# SPEC.md

## Project

Build and maintain a repeatable, non-leaking `SOXL` research and execution
harness.

This repository is not a general OpenClaw workspace and not a generic quant
sandbox. It is a `SOXL`-focused harness for developing, validating, persisting,
and operationalizing strategy behavior inside explicit boundaries.

## What The Harness Must Provide

The project is successful when the repository supports:

- reproducible training and validation
- explicit separation between signal generation, position sizing, broker
  execution, and reconciliation
- persisted artifacts that can be replayed and audited
- promotion or acceptance decisions backed by documented evidence
- a clean handoff from research outputs to runtime boundaries without one-off
  glue

Raw performance matters, but only inside a trustworthy harness.

## Scope

This repository is for `SOXL` only.

Out of scope:

- multi-ticker optimization
- undocumented ad-hoc experiments
- direct tuning on held-out `2022` labels outside the documented evaluation
  framework
- collapsing research, sizing, execution, and reconciliation into one opaque
  script

## System Boundaries

Keep these layers separate:

1. signal generation
2. position sizing
3. broker execution
4. reconciliation and monitoring

Code should remain structured accordingly:

- research and training: `src/soxl_quant/`
- execution adapters: `production/`
- execution workpads: `codex_exec_plans/`

## Harness Invariants

- The harness must be non-leaking by default.
- `2022` remains an evaluation window unless the split is explicitly redefined
  and documented.
- Any claim of improvement should include fixed-split evidence plus `2020`,
  `2022`, `2023-2024`, and walk-forward evidence when available.
- Artifact persistence, replayability, and traceability are first-class
  acceptance criteria.
- Chat history is not a source of truth.

## Repo-Owned Sources Of Truth

Use these layers intentionally:

- `AGENTS.md`: operating rules
- `WORKFLOW.md`: execution routing and completion gates
- `codex_exec_plans/active/` and `codex_exec_plans/completed/`: task history
  and proof of work
- `memory/`: continuity across sessions
- `skills/`: distilled reusable workflows and maintained operational knowledge

These roles should not blur together.

## Knowledge Distillation

This repo should get better as recurring work repeats.

- Stable context and important continuity belong in `memory/`.
- Live task state belongs in the active plan/workpad.
- Reusable methods, troubleshooting flows, and recurring guardrails belong in
  `skills/`.
- When a used skill diverges from current practice, update the skill as part of
  the work instead of leaving stale guidance behind.

## Current Working Model

The repository already contains the main harness pieces:

- qlib-native research and validation code
- artifact persistence
- runtime and paper-execution adapters
- repo-owned plan files under `codex_exec_plans/`

The active thread lives under `codex_exec_plans/active/`.

## Required Per-Iteration Output

Every substantial iteration should leave behind a durable record of:

- goal or hypothesis
- command or procedure
- evidence or failure signal
- keep / drop decision
- next action

If the work changes execution behavior, also record whether it changed:

- signal generation
- position sizing
- broker execution
- reconciliation or monitoring

## Guiding Principle

When a task reveals a weakness, prefer improving the harness itself:

- tighten validation
- clarify interfaces
- update repo-owned docs
- distill reusable practice into skills

The repository should accumulate leverage, not just outputs.
