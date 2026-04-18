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
- `.codex/skills/`: reusable methods and maintained operational knowledge. Codex CLI auto-loads skills from this path.
- `configs/soxl/profile_registry.yaml`: profile lifecycle, canonical baselines, and retention intent.
- `experiments/`: experiment-owned profile packages with colocated notes and expectations.
- `artifacts/retention_manifest.json`: generated cleanup ledger for model artifacts and repo noise.

Chat history is not the durable record.

## Memory

`memory/` is the continuity layer. Keep it narrow — key decisions, blockers, and
context worth carrying forward. Do not use memory as the main execution checklist
when the active plan should carry that state.

If you want future sessions to remember something important, write it down.

Use the generated `memory/index.json` and `scripts/memory_search.py` before
bulk-reading memory notes. Retrieval should start from the index, then open only
the top matching notes unless that first pass is insufficient.

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

When a loop changes profile lifecycle or retention:

- record the canonical comparison baseline
- record whether the profile is `canonical`, `candidate`, `ablation`, `dropped`, or `archived`
- record whether the actor bundle is meant to be kept or summary-only

## Profile Governance

Profile sprawl is now a harness problem, not just a naming problem.

- Every `configs/soxl/*.yaml` profile must have a row in `configs/soxl/profile_registry.yaml`.
- New active profile work should prefer `experiments/<status>/<name>/profile.yaml` plus colocated `notes.md` and `expectations.md`.
- Legacy flat files under `configs/soxl/*.yaml` are compatibility paths during migration, not the preferred place for new work.
- The registry is the source of truth for:
  - current canonical profile per family
  - baseline lineage
  - whether the actor bundle is retained
  - whether the summary is retained
- Before starting a new training branch, identify and write down:
  - the canonical baseline
  - the direct comparison baseline
  - whether the new profile is meant to replace, diagnose, or ablate
- A dropped profile keeps summary/review/memory evidence by default, but should not keep its actor checkpoint unless the registry explicitly says so.
- New profile files are not allowed to accumulate without a registry status and a notes reference.

## Code Boundaries

Do not let `cli.py`, `settings.py`, or `rl.py` become mixed-purpose catch-alls.

- `src/soxl_quant/settings.py` owns repo paths, constants, and artifact naming only.
- `src/soxl_quant/config.py` owns typed config dataclasses and profile loading.
- `src/soxl_quant/cli.py` owns orchestration only:
  - argument parsing
  - dataset/train/eval flow
  - walk-forward dispatch
  - persistence / promotion reporting
- RL helper logic must be split by concern:
  - execution rules
  - positioning / caps
  - observation construction
  - reward
  - policy / models
  - evaluation
- When a change alters one of those layers, keep the edit in that layer instead of patching another monolithic helper.

## Retention And Cleanup

Repository noise should be reduced continuously instead of in rare purges.

- Root-level ad-hoc analysis scripts are temporary. Either move them into `scripts/` with a clear reusable purpose or delete them.
- Generated caches such as `__pycache__` and `.DS_Store` should not survive task completion.
- Use `qlib_venv/bin/python scripts/prune_artifacts.py` to regenerate `artifacts/retention_manifest.json`.
- If cleanup is part of the task, prefer:
  - keep summaries, reviews, workpads, and memory notes
  - delete redundant actor checkpoints for dropped or archived profiles
  - keep only registry-approved actor bundles
- Do not delete large artifacts blindly; inspect the registry and manifest first.

## Skill Distillation & Auto-Optimization

`.codex/skills/` is where reusable method lives. Codex CLI auto-loads skills from this path.

- **Auto-skill rule**: After completing a complex task (5+ tool calls), fixing a
  tricky error, or discovering a non-trivial workflow, **evaluate and save** the
  approach as a skill (or update an existing skill). Do not turn ordinary session
  progress into a skill — distill the reusable workflow, guardrail, or
  troubleshooting method.
- **Auto-patch rule**: If a skill is used and found stale, incomplete, or wrong,
  **patch it immediately** as part of the task instead of leaving it drifting.
- **Optimization loop**: Every 3–5 related tasks, run the skill optimizer:
  1. **Inventory** — list active skills in `.codex/skills/`.
  2. **Deduplicate** — merge skills that overlap >50% in purpose or commands.
  3. **Refresh** — update each skill's front-matter and quick-start examples to
     match current repo state.
  4. **Commit** — treat the optimization as a task and commit the result.
- **Stale-skill rule**: If a skill has not been used in 30 days and has been
  superseded by another skill or repo change, delete it from `.codex/skills/`.

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

This workspace uses a single unified Python virtual environment:

- **`qlib_venv`** (Python 3.11) — Canonical environment for research,
  backtesting, and production execution. Contains Qlib, RL stack
  (`tianshou`, `torch`, `gymnasium`), analytics (`pandas`, `numpy`,
  `scikit-learn`), and broker adapters (`futu-api`).

All scripts in `scripts/`, `src/soxl_quant/`, and `production/` should run
against `qlib_venv`. Reproducible requirements are frozen in
`requirements/qlib.txt`.

**Goal for production:** Keep `qlib_venv` lean and well-versioned so it can be
recreated identically on the production host. Before adding new packages,
verify they do not conflict with existing pinned deps (especially
`protobuf`).

## Harness Engineering

This repository is engineered as a self-contained harness for autonomous agents.
These rules determine whether the agent can operate reliably at scale.

### Repository-Local Truth
All knowledge required to make engineering decisions must live inside this
repository. If a rule, pattern, or constraint exists only in human memory or
chat history, it does not exist for the agent.

### Mechanical Enforcement over Documentation
Prefer lint, tests, and executable scripts over written instructions.
- If a rule can be checked by code, it must be checked by code.
- If a rule is documented but not yet enforced mechanically, treat it as a
  temporary exception and open a plan to automate it.
- The agent must run the existing test suite before declaring any task complete.

### Executable Feedback Loops
Every task must end with a validation step that produces pass/fail evidence
the agent can read. Do not proceed based on assumption.
Examples: `qlib_venv/bin/python scripts/check_harness.py`, `python scripts/backtest_validate.py`, `make check`.

### Progressive Disclosure
`AGENTS.md` is a map, not a manual. Deep context lives in:
- `.codex/skills/` for reusable workflows
- `SPEC.md` for project intent and invariants
- `codex_exec_plans/active/` for the current execution context
- `docs/` for detailed design documentation
The agent should start here and navigate outward, not ingest everything at once.

### Intent as Code
A task without a written plan in `codex_exec_plans/active/` and explicit
acceptance criteria is not allowed to start. The plan is the primary interface
between human intent and agent execution.

### Continuous Refactoring
Agent-generated code drifts faster than human-written code. Every 3–5 tasks,
dedicate one iteration to harness maintenance: delete dead code, fix flaky
tests, consolidate duplicate scripts, and update stale skills.

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

Before starting a new training or replay loop:

1. query memory through `qlib_venv/bin/python scripts/memory_search.py ...`
2. validate the harness through `qlib_venv/bin/python scripts/check_harness.py`
3. inspect `artifacts/retention_manifest.json` before deleting or regenerating large artifacts

## Current Direction

The active thread lives under `codex_exec_plans/active/`.

Current repo intent:

- build a repeatable, non-leaking `SOXL` research and execution harness
- preserve clean boundaries between research outputs and runtime execution
- keep the harness maintainable as agent-generated complexity accumulates
