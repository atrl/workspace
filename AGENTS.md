# AGENTS.md

This repository is a `SOXL` harness workspace.

The job of the agent is to execute inside a repeatable research and validation
harness, leave evidence in the repository, and improve the harness when gaps are
exposed.

## Startup

Before doing substantial work:

1. Read `AGENTS.md`.
2. Read `SPEC.md`.
3. Read `docs/active_workpad.md` for the current active task context.
4. Read `runtime/<direction>/knowledge.md` for continuity in the chosen direction.

`WORKFLOW.md` is an external orchestration template (Symphony/Linear). Read it only when
using the Symphony automation layer; it is not part of the agent's core startup.

Do not ask permission for this startup sequence.

## Sources Of Truth

Use these files for distinct roles. Do not collapse them together.

- `AGENTS.md`: top-level operating contract.
- `SPEC.md`: project intent, invariants, and boundaries.
- `WORKFLOW.md`: execution routing and state rules.
- `docs/active_workpad.md`: the live workpad for the current thread.
- `docs/research_knowledge/index.md`: first-stop task direction router.
- `docs/research_knowledge/task_router.json`: machine-readable mapping from
  task direction to knowledge file, skill, runtime query path, and
  validation commands.
- `docs/research_knowledge/`: mechanism-specific research conclusions split by
  factor, data, model selection, network, reward, execution, and multi-asset
  lines.
- `runtime/<direction>/knowledge.md`: continuity across sessions, organized by task direction.
- `.codex/skills/`: reusable methods and maintained operational knowledge. Codex CLI auto-loads skills from this path.
- `runtime/harness/card/registry/profile.yaml`: profile lifecycle registry, canonical baselines, and retention intent.
- `experiments/`: experiment-owned profile packages with colocated notes and expectations.
- `data/datasets.yaml`: generated data registry for raw symbols, processed
  panels, qlib stores, and training caches.
- `runtime/harness/card/registry/retention_manifest.json`: generated cleanup ledger for model artifacts and repo noise.

Chat history is not the durable record.

## Task Direction Routing

Before reading broad runtime knowledge or starting implementation, classify the task into
exactly one primary direction:

- `factor`: feature hypotheses, IC checks, rulebase-derived signals, signal backtests.
- `data`: raw CSVs, qlib stores, processed panels, training caches, external data.
- `model_selection`: checkpoints, fixed splits, promotion gates, actor retention.
- `network`: GRU, attention, MoE, mode classifiers, asset-specific heads.
- `reward`: Kelly, penalties, imitation, activity, opportunity cost, trade quality.
- `execution`: fixed-actor overlays, position budgets, mode routing, dashboards.
- `multi_asset`: TQQQ/UPRO/SPXL/TECL transfer tests or shared 3x ETF training.
- `production`: broker adapters, paper/live execution, reconciliation, monitoring.
- `harness`: repo structure, validation, cleanup, docs, skills, automation.

Use `docs/research_knowledge/task_router.json` for research directions. The
route determines:

1. which `docs/research_knowledge/*.md` file to read,
2. which `.codex/skills/*` workflow to invoke,
3. which `runtime/<direction>/knowledge.md` to query for historical context,
4. which `runtime/<category>/` paths can be opened first,
5. which validation commands prove the task.

`runtime/<direction>/knowledge.md` is not a starting point. It is evidence retrieval behind the selected
direction. Do not bulk-read all knowledge files before choosing the route.

## Runtime Knowledge

`runtime/<direction>/knowledge.md` is the continuity layer. It accumulates key
decisions, blockers, and context worth carrying forward per direction.
Do not use it as the main execution checklist when the active workpad should
carry that state.

If you want future sessions to remember something important, write it into the
relevant `runtime/<direction>/knowledge.md` or the experiment's `notes.md`.

Knowledge retrieval must be scoped by task direction. Example:

- factor work: read `runtime/factor/knowledge.md` then open `runtime/factor/`
- model-selection work: read `runtime/model/knowledge.md` then open `runtime/model/`
- execution work: read `runtime/execution/knowledge.md` then open `runtime/execution/`

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

Before starting any new experiment loop, the active workpad must also state:

- closest historical success case
- closest historical failure case
- the new mechanism being tested this time
- the row that will be added back into the comparison table

When a loop changes profile lifecycle or retention:

- record the canonical comparison baseline
- record whether the profile is `canonical`, `candidate`, `ablation`, `dropped`, or `archived`
- record whether the actor bundle is meant to be kept or summary-only

## Profile Governance

Profile sprawl is now a harness problem, not just a naming problem.

- Every profile yaml (found under `runtime/<direction>/card/<name>/profile.yaml`) must have a row in `runtime/harness/card/registry/profile.yaml`.
- New active profile work should prefer `experiments/<status>/<name>/profile.yaml` plus colocated `notes.md` and `expectations.md`.
- Legacy flat files under `runtime/<direction>/card/<name>/profile.yaml` are the canonical profile location; `experiments/<status>/<name>/profile.yaml` is preferred for new work.
- The registry is the source of truth for:
  - current canonical profile per family
  - baseline lineage
  - whether the actor bundle is retained
  - whether the summary is retained
- Before starting a new training branch, identify and write down:
  - the canonical baseline
  - the direct comparison baseline
  - whether the new profile is meant to replace, diagnose, or ablate
- A dropped profile keeps summary/review/continuity evidence by default, but should not keep its actor checkpoint unless the registry explicitly says so.
- New profile files are not allowed to accumulate without a registry status and a notes reference.

## Experiment Entry Gate

New experiments are not allowed to start from "maybe this knob helps".

Before any new training, replay, or profile branch is created, write down:

- canonical baseline
- direct comparison baseline
- experiment role:
  - `replace`
  - `diagnose`
  - `ablate`
- target improvement window:
  - for example `test_2022`, restart windows, `test_2023_2024`, or walk-forward
- accepted trade-off window:
  - what may get slightly worse without invalidating the run
- changed layer:
  - `feature`
  - `network`
  - `model`
  - `reward`
  - `execution`
  - `selection`
  - `sampling`
  - `data`

If those fields are missing, the experiment is not ready to run.

## External-Evidence Requirement

Future experiments must expand the search space, not rename old loops.

Every new experiment must cite at least one of:

- reference paper
- open-source project
- new dataset
- new feature family
- new network architecture
- new model-selection method
- new reward-function design

The reference must be written into the experiment notes or active workpad before the run starts.

Repeating a prior branch with only threshold or coefficient drift is not enough unless the workpad explains:

- which historical branch it is most similar to
- why that branch failed or succeeded
- what concrete new mechanism now exists

## Anti-Repetition Rule

Exploration breadth is now a harness constraint.

Before starting a new experiment, query the repo history and answer:

1. what is the closest historical success case
2. what is the closest historical failure case
3. why is this run not a duplicate of either

If the difference is only:

- a stronger penalty
- a slightly different threshold
- a renamed profile
- a larger bundle of already-failed levers

then default to not running it.

## Experiment Card Contract

Every new experiment package or notes file must contain:

- motivation
- external reference
- hypothesis
- target slices / windows
- acceptance criteria
- expected trade-off
- failure mode
- comparison-table row key

The implementer should not need to guess what success means.

## Comparison Table Discipline

The repository must maintain a single durable comparison surface for historical conclusions.

- Major conclusions belong in `docs/experiment-landscape-review.md`.
- Pairwise or run-local reviews can still live under `runtime/model/`.
- A training or replay loop is not complete until its conclusion is reflected in the durable comparison record.

Every completed experiment must add or update a row containing:

- profile name
- category
- baseline
- mechanism delta
- key metrics
- keep / drop / candidate decision
- next action

If a run is not important enough to earn a row in the comparison table, it is probably not important enough to run.

## Pre-Run Reflection

Before the next experiment begins, review both:

- where past strategies were clearly strong
- where past strategies were clearly weak

At minimum, write down:

- the success windows to preserve
- the failure windows to improve
- why the new proposal should improve the weak windows without erasing the strong ones

This reflection must live in the workpad or experiment notes, not only in chat.

## Code Boundaries

Do not let `cli.py`, `settings.py`, or `rl.py` become mixed-purpose catch-alls.

- Preferred code layout under `src/soxl_quant/` is now:
  - `app/` for orchestration entrypoints
  - `core/` for config, registry, constants, artifacts, and dataset helpers
  - `policy/` for observation, reward, execution, positioning, evaluation, and model builders
- `src/soxl_quant/settings.py` owns repo paths, constants, and artifact naming only.
- `src/soxl_quant/config.py` owns typed config dataclasses and profile loading.
- `src/soxl_quant/app/` is the primary orchestration surface:
  - `cli.py`: command shell only
  - `bootstrap.py`: runtime init, dataset bootstrap, multi-asset bootstrap wiring
  - `train.py`: profile-run orchestration
  - `selection.py`: staged training and checkpoint selection
  - `sources.py`: multi-asset source preparation
  - `workers.py`: walk-forward subprocess orchestration
  - `evaluate.py`: validation scoring and slice helpers
  - `promote.py`: promotion gate evaluation
  - `reporting.py`: end-of-run result persistence and summary printing
  - `artifacts.py`: actor bundle and summary persistence helpers
- `src/soxl_quant/cli.py` is a compatibility facade, not the preferred place for new orchestration code.
- RL helper logic must be split by concern:
  - execution rules
  - positioning / caps
  - observation construction
  - reward
  - policy / models
  - evaluation
- `src/soxl_quant/rl.py` is a legacy compatibility facade only. New code, tests, and repo-owned scripts should import from `src/soxl_quant/policy/` directly.
- When a change alters one of those layers, keep the edit in that layer instead of patching another monolithic helper.

## Script Boundaries

Repo-owned scripts are treated as internal code, not external clients.

- New or modified files under `scripts/` should import from `src/soxl_quant/app/`, `src/soxl_quant/policy/`, `src/soxl_quant/core/`, or `src/soxl_quant/runtime.py` directly.
- Do not point new script work at `src/soxl_quant/rl.py` or `src/soxl_quant/cli.py` unless the task is explicitly about compatibility.
- If a script still depends on a compatibility facade, that is refactor debt and should be cleaned up as part of the next related task.

## Directory Read Order

When orienting to the repo, prefer indexes and summaries before raw outputs.

- `experiments/`:
  - read `experiments/index.md`
  - then `experiments/README.md`
  - then `profile.yaml`
  - then `expectations.md`
  - then `notes.md`
- `docs/research_knowledge/`:
  - read `docs/research_knowledge/index.md`
  - read `docs/research_knowledge/task_router.json`
  - then the one mechanism file relevant to the proposed change
- `runtime/`:
  - read `runtime/README.md`
  - then `runtime/index.json`
  - prefer summary markdown / JSON over raw logs and CSV first
- `runtime/<direction>/knowledge.md`:
  - read after choosing a task direction
  - then open the corresponding `runtime/<category>/` paths
  - prefer summary markdown over raw logs and CSV first

## Compatibility Policy

Compatibility facades exist to avoid breaking old entrypoints while the harness is being refactored.

- Current compatibility surfaces:
  - `src/soxl_quant/cli.py`
  - `src/soxl_quant/rl.py`
- New behavior must not be added to those files unless the task is specifically compatibility maintenance.
- Every few refactor loops, shrink the exported surface of compatibility modules to the minimum still required by external callers.

## Retention And Cleanup

Repository noise should be reduced continuously instead of in rare purges.

- Root-level ad-hoc analysis scripts are temporary. Either move them into `scripts/` with a clear reusable purpose or delete them.
- Generated caches such as `__pycache__` and `.DS_Store` should not survive task completion.
- Use `qlib_venv/bin/python scripts/prune_artifacts.py` to regenerate `runtime/harness/card/registry/retention_manifest.json`.
- If cleanup is part of the task, prefer:
  - keep summaries, reviews, workpads, and runtime knowledge notes
  - delete redundant actor checkpoints for dropped or archived profiles
  - keep only registry-approved actor bundles
- Do not delete large artifacts blindly; inspect the registry and manifest first.
- `__pycache__/` and stale generated files under `src/soxl_quant/` are cleanup debt; remove them when they are encountered as part of nearby refactor or hygiene work.

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
failure — and record the reason in runtime/harness/knowledge.md if so.

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
- If a research workflow rule is repeated more than twice in docs or runtime knowledge files, plan to enforce it mechanically through scripts, templates, or checks.

### Executable Feedback Loops
Every task must end with a validation step that produces pass/fail evidence
the agent can read. Do not proceed based on assumption.
Examples: `qlib_venv/bin/python scripts/check_harness.py`, `python scripts/backtest_validate.py`, `make check`.

### Progressive Disclosure
`AGENTS.md` is a map, not a manual. Deep context lives in:
- `.codex/skills/` for reusable workflows
- `SPEC.md` for project intent and invariants
- `docs/active_workpad.md` for the current execution context
- `docs/` for detailed design documentation
The agent should start here and navigate outward, not ingest everything at once.

### Intent as Code
A task without a written plan in `docs/active_workpad.md` and explicit
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

See `SPEC.md` for scope, system boundaries, and layer separation.

Quick reference:
- research and training: `src/soxl_quant/`
- execution adapters: `production/`
- workpads and decision trail: `docs/active_workpad.md`
- data and analysis scripts: `scripts/`
- unified runtime state + config: `runtime/`

`runtime/` is the single writable surface for both configuration and generated state.

## Worktree Policy

Use git worktrees only for **short-lived experimental branches** that need isolation from `main`.

- **Location:** `.worktrees/<branch-name>/` (already ignored by `.gitignore`)
- **Naming:** match the branch name exactly
- **Lifecycle:**
  1. Create worktree when starting a substantial experiment branch
  2. Run the experiment, collect evidence
  3. Merge or cherry-pick conclusions back to `main`
  4. Delete the worktree immediately after the branch is merged or abandoned
- **Do not** use worktrees for long-term development or as a secondary workspace
- **Do not** commit worktree contents to the repository (`.worktrees/` is in `.gitignore`)
- Before creating a worktree, verify `.worktrees/` is ignored: `git check-ignore -q .worktrees`

## Execution Standard

For every substantial iteration:

1. ground in the current repo state
2. update or confirm the active workpad
3. run exactly one meaningful loop
4. validate the result
5. write proof of work back to the repo
6. update continuity in `runtime/<direction>/knowledge.md` or the experiment's `notes.md`
7. **commit and push the changes**
8. if a skill was created or modified, verify `.codex/skills/` is synced

If a run fails, document the failure and tighten the harness instead of blindly
retrying.

Before starting a new training or replay loop:

1. query historical context through the relevant `runtime/<direction>/knowledge.md`
2. validate the harness through `qlib_venv/bin/python scripts/check_harness.py`
3. inspect `runtime/harness/card/registry/retention_manifest.json` before deleting or regenerating large artifacts

When creating a new experiment profile:

1. create `runtime/<direction>/card/<name>/profile.yaml` with the full configuration
2. add a row to `runtime/harness/card/registry/profile.yaml` with baseline, status, and notes_ref
3. colocate experiment notes in `runtime/<direction>/knowledge.md` or a dedicated note under the card directory
4. do not create a separate top-level `configs/` or `experiments/` directory for the profile

## Current Direction

The active thread lives under `docs/active_workpad.md`.

Current repo intent:

- build a repeatable, non-leaking `SOXL` research and execution harness
- preserve clean boundaries between research outputs and runtime execution
- keep the harness maintainable as agent-generated complexity accumulates
