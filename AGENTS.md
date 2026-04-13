# AGENTS.md

This repository is a `SOXL` harness workspace.

The job of the agent in this repo is not to improvise a personal workflow. The
job is to execute inside a repeatable research and validation harness, leave
evidence in the repository, and improve the harness when gaps are exposed.

## Startup

Before doing substantial work:

1. Read `SOUL.md`.
2. Read `USER.md`.
3. Read `memory/YYYY-MM-DD.md` for today and yesterday if those files exist.
4. Read `MEMORY.md` only in a private main session, and only if it exists.
5. Read `AGENTS.md`.
6. Read `SPEC.md`.
7. Read the active workpad under `codex_exec_plans/active/`.

Do not ask permission for this startup sequence.

## Sources Of Truth

Use these files for distinct roles. Do not collapse them together.

- `AGENTS.md`: top-level operating contract.
- `SPEC.md`: project intent, invariants, and boundaries.
- `WORKFLOW.md`: execution routing and state rules.
- `codex_exec_plans/active/`: the live workpad for the current thread.
- `codex_exec_plans/completed/`: completed or abandoned threads.
- `memory/`: continuity across sessions.
- `skills/`: reusable methods and maintained operational knowledge.

Chat history is not the durable record.

## Memory

`memory/` is the continuity layer. Keep it, but keep it narrow.

- `memory/YYYY-MM-DD.md`: session outcomes, decisions, blockers, and context
  worth carrying forward.
- `MEMORY.md`: curated long-term preferences or enduring context if it exists.

Do not use memory as the main execution checklist when the active plan should
carry that state.

If you want future sessions to remember something important, write it down.

## Workpad Discipline

The active plan file is the primary execution record.

- Reuse the existing active workpad instead of scattering status across random
  notes.
- Keep the workpad current with goal, acceptance criteria, commands, evidence,
  decisions, and next action.
- Capture baseline or reproduction evidence before changing code when the task
  depends on a behavior change.
- A task is not complete until the repository contains the proof of work.

Every substantial loop should leave behind:

- goal or hypothesis
- command or procedure
- validation evidence or failure evidence
- keep / drop / next-action decision

## Skill Distillation

`skills/` is where reusable method goes.

- After a complex task, tricky fix, or non-trivial workflow discovery, evaluate
  whether the method should become a skill or update an existing skill.
- If a skill is used and found stale, incomplete, or wrong, patch it as part of
  the task instead of leaving it drifting.
- Do not turn ordinary session progress into a skill. Distill the reusable
  workflow, guardrail, or troubleshooting method.

## Default Posture

- Operate autonomously inside this repository.
- Prefer improving shared harness paths over creating one-off scripts.
- Prefer explicit interfaces, configs, and validation over hidden conventions.
- When a failure exposes a missing guardrail, fix the harness, not just the
  immediate symptom.
- Keep unrelated repo churn out of scope.

## Red Lines

- Do not exfiltrate private data.
- Do not run destructive commands without asking.
- Do not rely on chat history as the sole record of important work.
- Do not create a parallel workflow outside the repo-owned documents.

## Project Boundaries

This workspace is for `SOXL` only.

Keep these layers separate:

1. signal generation
2. position sizing
3. broker execution
4. reconciliation and monitoring

Keep code in the intended locations:

- research and training: `src/soxl_quant/`
- execution adapters: `production/`
- workpads and decision trail: `codex_exec_plans/`

## Harness Invariants

- Do not optimize directly on held-out `2022` labels unless the split is
  explicitly redefined and documented.
- Treat `2022` as an evaluation window by default.
- Any claim of improvement should include the fixed split, `2020`, `2022`,
  `2023-2024`, and walk-forward evidence when available.
- Artifact persistence, replayability, and validation traceability matter as
  much as raw returns.

## Execution Standard

For every substantial iteration:

1. ground in the current repo state
2. update or confirm the active workpad
3. run exactly one meaningful loop
4. validate the result
5. write proof of work back to the repo
6. update continuity in `memory/`

If a run fails, document the failure and tighten the harness instead of blindly
retrying.

## Current Direction

The active thread lives under `codex_exec_plans/active/`.

Current repo intent:

- build a repeatable, non-leaking `SOXL` research and execution harness
- preserve clean boundaries between research outputs and runtime execution
- keep the harness maintainable as agent-generated complexity accumulates
