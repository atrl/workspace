# AGENTS.md

This repository is a `SOXL` harness workspace.

The job of the agent is to execute inside a repeatable research and validation
harness, leave evidence in the repository, and improve the harness when gaps are
exposed.

## Startup

Before doing substantial work:

1. Read `AGENTS.md`.
2. Read `SPEC.md`.
3. Read `WORKFLOW.md`.
4. Read the active workpad under `codex_exec_plans/active/`.
5. Read `memory/` for any continuity worth carrying forward.

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
- `.codex/skills/`: symlinked skills that Codex CLI auto-loads for this workspace.

Chat history is not the durable record.

## Memory

`memory/` is the continuity layer. Keep it narrow — key decisions, blockers, and
context worth carrying forward. Do not use memory as the main execution checklist
when the active plan should carry that state.

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

## Skill Distillation & Auto-Optimization

`skills/` is where reusable method goes. `.codex/skills/` mirrors the active
skills so Codex CLI can auto-load them in this workspace.

- **Auto-skill rule**: After completing a complex task (5+ tool calls), fixing a
  tricky error, or discovering a non-trivial workflow, **evaluate and save** the
  approach as a skill (or update an existing skill). Do not turn ordinary session
  progress into a skill — distill the reusable workflow, guardrail, or
  troubleshooting method.
- **Auto-patch rule**: If a skill is used and found stale, incomplete, or wrong,
  **patch it immediately** as part of the task instead of leaving it drifting.
- **Optimization loop**: Every 3–5 related tasks, run the skill optimizer:
  1. **Inventory** — list active skills in `skills/` (ignore `archive/`).
  2. **Deduplicate** — merge skills that overlap >50% in purpose or commands.
  3. **Refresh** — update each skill's front-matter and quick-start examples to
     match current repo state.
  4. **Re-index** — ensure `.codex/skills/` symlinks reflect the active skill set.
  5. **Commit** — treat the optimization as a task and commit the result.
- **Stale-skill rule**: If a skill has not been used in 30 days and has been
  superseded by another skill or repo change, move it to `skills/archive/` and
  remove its symlink from `.codex/skills/`.

## Auto-Commit Discipline

Every task that changes files in the repo **must** end with:

1. `git add` of all relevant changes.
2. `git commit` with a descriptive message.
3. `git push` to sync evidence to the remote.

Only skip push when there is a confirmed lack of write permission or network
failure — and record the reason in memory if so.

## Default Posture

- Operate autonomously inside this repository.
- Prefer improving shared harness paths over creating one-off scripts.
- Prefer explicit interfaces, configs, and validation over hidden conventions.
- When a failure exposes a missing guardrail, fix the harness, not just the
  immediate symptom.
- Keep unrelated repo churn out of scope.

## Environment

This workspace uses two Python virtual environments:

- **`qlib_venv`** (Python 3.11) — Primary research environment. Use for Qlib,
  RL training, backtesting, and anything in `src/soxl_quant/` or `scripts/`.
- **`futu_venv`** (Python 3.14) — Broker-specific environment for Futu API
  scripts in `production/`. Kept separate because some Futu dependencies have
  not been validated on 3.11.

**Consolidation rule:** New packages should default to `qlib_venv`. Only add
packages to `futu_venv` when they are strictly required by broker scripts and
fail in `qlib_venv`. Reproducible requirements are frozen under
`requirements/`.

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
- data and analysis scripts: `scripts/`

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
7. **commit and push the changes**
8. if a skill was created or modified, verify `.codex/skills/` is synced

If a run fails, document the failure and tighten the harness instead of blindly
retrying.

## Current Direction

The active thread lives under `codex_exec_plans/active/`.

Current repo intent:

- build a repeatable, non-leaking `SOXL` research and execution harness
- preserve clean boundaries between research outputs and runtime execution
- keep the harness maintainable as agent-generated complexity accumulates
