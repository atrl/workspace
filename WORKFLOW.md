# WORKFLOW.md

This file is a thin orchestration adapter for external workflow runners.
It exists so tools such as Symphony can enter this repository without inventing a second operating contract.

The source of truth for how work is done in this repo remains:

- `AGENTS.md`
- `SPEC.md`
- the active plan in `codex_exec_plans/active/`

If this file conflicts with `AGENTS.md`, follow `AGENTS.md`.

## Purpose

Use this workflow only to:

- bootstrap an unattended repo session
- route work into the repo-owned contract
- make state transitions explicit for orchestration tools

Do not duplicate detailed project rules here.
Do not create a parallel `agent.md` process.

## Repository Intent

This repository is for `SOXL` only.

Primary objective:

- build and operationalize a repeatable, non-leaking `qlib`-native `SOXL` research and execution workflow

Read first for intent and boundaries:

- `SPEC.md`

Read first for operating rules:

- `AGENTS.md`

## Startup Sequence

Before doing any substantial work:

1. Read `SOUL.md`
2. Read `USER.md`
3. Read `memory/YYYY-MM-DD.md` for today and yesterday
4. If this is a direct private session with the human, also read `MEMORY.md` if it exists
5. Read `AGENTS.md`
6. Read `SPEC.md`
7. Read `HEARTBEAT.md` if this is a heartbeat/background loop
8. Read the active file under `codex_exec_plans/active/`

Do not ask permission for this startup sequence.

## Current Execution Entry Point

The active execution thread is the file currently under:

- `codex_exec_plans/active/`

At the time this workflow was added, the active plan is:

- `codex_exec_plans/active/2026-04-10-soxl-agent-loop.md`

If the active file changes later, follow the new active file instead of this hardcoded example.

## Operating Loop

For each substantial unattended run, execute exactly one meaningful loop:

- one validation run, or
- one harness improvement, or
- one production-layer improvement with verification

Before starting:

- inspect runtime state
- do not start overlapping training, replay, or validation runs that could corrupt artifacts or logs

After finishing:

- write proof of work back to the repo
- append the result to the active plan
- add the important part to `memory/YYYY-MM-DD.md`
- move finished or abandoned plans into `codex_exec_plans/completed/` when appropriate

## Required Proof Of Work

A loop is not complete until the repo contains the result.

Every substantial iteration should leave behind:

- hypothesis
- command or procedure executed
- result metrics or failure evidence
- keep / drop / next-action decision

If work touches execution or deployment, also record whether it changed:

- signal generation
- position sizing
- broker execution
- reconciliation or monitoring

## State Model

This repo does not require a specific external tracker.
If an external orchestration system needs states, use this minimal mapping:

- `Todo`: work is queued but not started
- `In Progress`: active implementation or validation is underway
- `Blocked`: true external blocker such as missing auth, missing secret, or missing required external tool
- `Review`: code or results are ready for human review
- `Done`: loop is finished and proof of work is written back to the repo

Only mark `Done` when the repo has been updated with proof of work.

## Default Posture

- Operate autonomously inside this repository
- Prefer implementation and verification over discussion
- Do not ask the human to perform routine follow-up actions
- Stop early only for a true blocker
- Keep work inside this repo; do not touch unrelated paths

## Boundaries

Preserve the repo's explicit layer boundaries:

1. Signal generation
2. Position sizing
3. Broker execution
4. Reconciliation and monitoring

Keep code in the intended locations:

- research and training: `src/soxl_quant/`
- execution adapters: `production/`
- experiment records: `codex_exec_plans/`

## Non-Leak Rules

- Do not optimize directly on held-out `2022` labels unless the split is explicitly redefined and documented
- Treat `2022` as an evaluation window by default
- Any claim of improvement should include:
  - fixed split result
  - `2020` slice
  - `2022` slice
  - `2023-2024` slice
  - walk-forward evidence when available

## Final Reporting

For unattended orchestration sessions, final reporting should contain:

- completed actions
- validation or failure evidence
- blockers, if any

Do not treat chat history as the durable record.
The durable record is the repository itself.
