# SPEC.md

## Project

Build and operationalize a `qlib`-native trading workflow for `SOXL` only.

This repository is not a general quant sandbox anymore.
Its single active target is:

- `SOXL` signal generation
- `SOXL` position sizing
- `SOXL` paper-execution readiness

## Objective

Produce a repeatable, non-leaking SOXL research and execution harness that:

- beats the old qlib baseline on return or risk-adjusted quality
- materially improves stressed-regime behavior, especially `2022`
- preserves traceability from idea to validation to deployment boundary
- supports replay and paper execution without rewriting the research path

## Current State

The repository already has the core structure for the intended lifecycle:

- research and training:
  - `src/soxl_quant/`
- execution adapters and runners:
  - `production/`
- experiment harness and decision trail:
  - `codex_exec_plans/`

Documented current progress:

- actor artifact persistence exists
- paper replay exists
- hard drawdown capping materially reduced chronic `2022` overexposure
- walk-forward and balanced runs still suffer from qlib shutdown instability under Python `3.14`

The current active thread is:

- `codex_exec_plans/active/2026-04-10-soxl-agent-loop.md`

## Non-Goals

Do not use this repository for:

- multi-ticker optimization beyond `SOXL`
- undocumented ad-hoc experiments
- tuning directly on held-out `2022` labels
- collapsing signal generation, sizing, broker execution, and reconciliation into one opaque script

## System Boundaries

Keep these layers separate:

1. Signal generation
2. Position sizing
3. Broker execution
4. Reconciliation and monitoring

Research code should live under `src/soxl_quant/`.
Execution adapters should live under `production/`.
Experiment records should live under `codex_exec_plans/`.

## Success Criteria

A strategy direction is only considered promising if it leaves behind documented evidence including:

- fixed-split result
- `2020` slice
- `2022` slice
- `2023-2024` slice
- walk-forward evidence when available

Operational progress is only considered real if:

- artifacts can be persisted and replayed
- commands can be reproduced from repo docs
- paper execution can consume the model output without custom one-off glue

## Primary Risks

- leakage through sampling, splits, or stress-window selection
- overfitting toward return while degrading drawdown quality
- policy behavior that remains overexposed during drawdowns
- qlib lifecycle instability on Python `3.14`
- undocumented experiments that cannot be audited later

## Workflow Contract

This repo follows a Symphony-inspired, repo-owned workflow model.
That means:

- the repository contains the operating contract
- plans are explicit and versioned
- each substantial task leaves proof of work in the repo
- chat history is not the source of truth

Top-level operating contract:

- `AGENTS.md`

Execution harness:

- `codex_exec_plans/README.md`
- active plan in `codex_exec_plans/active/`
- completed plans in `codex_exec_plans/completed/`

Heartbeat / background loop contract:

- `HEARTBEAT.md`

## Required Per-Iteration Output

Every substantial iteration should record:

- hypothesis
- command or procedure
- result metrics or failure evidence
- keep / drop decision
- next action

If the work is implementation-heavy rather than experiment-heavy, record:

- what changed
- how it was validated
- what boundary it affects:
  - signal
  - sizing
  - execution
  - reconciliation

## Immediate Priorities

1. Keep the hard cap with floor `0.4` as the documented working baseline until replaced by a better validated result.
2. Tune cap activation timing to improve `2023-2024` behavior without losing the `2022` defense.
3. Isolate or eliminate qlib shutdown failures so walk-forward runs complete cleanly.
4. Extend broker-adapter work without coupling it to actor training or replay logic.

## How To Use This File

Read `SPEC.md` to understand what the repository is trying to achieve.
Read `AGENTS.md` to understand how work must be carried out.
Read the active file in `codex_exec_plans/active/` to understand the current thread and next experiment.
