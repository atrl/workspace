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

- `docs/research_knowledge/`: mechanism-specific research conclusions and task routing.
- `runtime/<direction>/knowledge.md`: continuity across sessions, organized by task direction.
- `.codex/skills/`: reusable methods and operational knowledge.
- `runtime/harness/card/registry/profile.yaml`: profile registry and retention intent.


Chat history is not the durable record.

## Task Direction Routing

- `factor`, `data`, `model_selection`, `network`, `reward`, `execution`, `multi_asset`, `production`, `harness`



## Runtime Knowledge

`runtime/<direction>/knowledge.md` is the continuity layer. It accumulates key
decisions, blockers, and context worth carrying forward per direction.
Do not use it as the main execution checklist when the active workpad should
carry that state.

If you want future sessions to remember something important, write it into the
relevant `runtime/<direction>/knowledge.md` or the experiment's `notes.md`.

Knowledge retrieval must be scoped by task direction.

## Workpad Discipline

- Reuse the existing active workpad instead of scattering status across random
  notes.
- Keep the workpad current for active cross-session tasks, but do not add
  single-run experiment conclusions there by default. For one-off experiments,
  put the conclusion in the corresponding runtime notebook instead.
- Capture baseline or reproduction evidence before changing code when the task
  depends on a behavior change.
- A task is not complete until the repository contains the proof of work.

## Notebook Experiment Discipline

- Future SOXL research, diagnostics, backtests, broker-data checks, and model
  iteration should go to `ipynb` by default. Treat the user's standing
  preference as: "尽量都到 ipynb" unless the task is purely code maintenance,
  operational startup, or explicitly asks for another artifact.
- Notebook-first is the default for research work. When a task is an experiment,
  diagnostic, factor screen, model/reward/network trial, broker-data audit,
  backtest, parameter screen, or one-off analysis, put the executable workflow
  and conclusion in a runtime notebook unless the user explicitly asks
  otherwise.
- Default to an executable notebook under
  `runtime/<direction>/card/<experiment>/<experiment>.ipynb` for experiments,
  diagnostics, parameter screens, backtests, and one-off research loops.
- Put single-run special-case logic, metrics, comparisons, conclusions, and
  artifact links in that notebook. The notebook should be runnable evidence, not
  just a narrative summary.
- Use a non-notebook artifact for experiment work only when the task is not an
  experiment/research loop, notebook tooling is unavailable, or the user
  explicitly asks for another format.
- Keep reusable abstractions in `src/soxl_quant/`; keep scripts thin and
  reusable. Do not push experiment-specific logic into shared modules unless it
  becomes a reusable interface with tests.
- Do not update `docs/active_workpad.md`, `runtime/<direction>/knowledge.md`,
  `docs/research_knowledge/`, `docs/experiment-landscape-review.md`, or
  `.codex/skills/` with single-run conclusions unless the user explicitly asks,
  the result is promoted, or the finding changes durable cross-experiment
  routing/governance.
- If a task needs both reusable code and experiment-specific logic, keep the
  reusable code in `src/` with tests and keep the special-case orchestration,
  parameter choices, outputs, and conclusion in the runtime notebook.
- If a notebook writes sidecar Markdown/JSON/CSV artifacts, the notebook remains
  the primary human-readable conclusion for that run.
- When in doubt, choose a runtime notebook first. Use Markdown, JSON, or CSV as
  sidecar evidence only when it makes the notebook more reproducible or
  machine-readable.
- Avoid creating Markdown-only single-run reports when an executable notebook
  can hold the same evidence. Markdown is acceptable as a promoted summary,
  registry note, or sidecar, but not as the default primary artifact.



## Profile Governance

- Every profile yaml must have a registry row in `runtime/harness/card/registry/profile.yaml`.

- `experiments/<status>/<name>/profile.yaml` is preferred for new work; legacy profiles live under `runtime/<direction>/card/<name>/profile.yaml`.
- The registry is the source of truth for canonical profiles, baseline lineage, and retention intent.
- Before starting a new training branch, identify and write down:
  - the canonical baseline
  - the direct comparison baseline
  - whether the new profile is meant to replace, diagnose, or ablate
- A dropped profile keeps summary/review/continuity evidence by default, but should not keep its actor checkpoint unless the registry explicitly says so.
- New profile files are not allowed to accumulate without a registry status and a notes reference.

## Experiment Governance

A new experiment is not ready to run until the runtime experiment notebook
(preferred) or notes file contains:

- **Baselines**: canonical and direct comparison
- **Role**: `replace`, `diagnose`, or `ablate`
- **Target window** (e.g. `test_2022`, walk-forward) and **accepted trade-off**
- **Changed layer**: `feature`, `network`, `model`, `reward`, `execution`, `selection`, `sampling`, or `data`
- **External reference**: paper, project, dataset, feature family, architecture, model-selection method, or reward design
- **Hypothesis, acceptance criteria, expected trade-off, failure mode**
- **Comparison-table row key**

Before starting, query repo history and answer:
1. closest historical success case
2. closest historical failure case
3. why this run is not a duplicate of either

Default to **not running** if the difference is only a renamed profile, a tweaked threshold, or a stronger penalty on an already-failed lever. After completion, keep the single-run conclusion in the runtime notebook. Add a row to `docs/experiment-landscape-review.md` only for promoted results, durable cross-experiment decisions, or explicit user requests.

## Code Boundaries

- `src/soxl_quant/app/` for orchestration, `core/` for config/helpers, `policy/` for RL logic.
- `src/soxl_quant/settings.py` owns repo paths, constants, and artifact naming only.
- `src/soxl_quant/config.py` owns typed config dataclasses and profile loading.
- `src/soxl_quant/cli.py` is a compatibility facade; do not add new orchestration code there.
- RL helper logic must be split by concern (execution, positioning, observation, reward, policy, evaluation).
- `src/soxl_quant/rl.py` is a legacy compatibility facade only. New code, tests, and repo-owned scripts should import from `src/soxl_quant/policy/` directly.


## Script Boundaries

- New or modified files under `scripts/` should import from `src/soxl_quant/app/`, `src/soxl_quant/policy/`, `src/soxl_quant/core/`, or `src/soxl_quant/runtime.py` directly.
- Do not point new script work at `src/soxl_quant/rl.py` or `src/soxl_quant/cli.py` unless the task is explicitly about compatibility.


## Directory Read Order

When orienting to the repo, prefer indexes and summaries before raw outputs.

- `experiments/`: index → README → profile.yaml → expectations → notes
- `docs/research_knowledge/`: index → task_router.json → relevant mechanism file
- `runtime/`: README → index.json → summaries before raw logs/CSV
- `runtime/<direction>/knowledge.md`: read after choosing a direction, then open the corresponding paths

## Compatibility Policy

- Current compatibility surfaces:
  - `src/soxl_quant/cli.py`
  - `src/soxl_quant/rl.py`
- New behavior must not be added to those files unless the task is specifically compatibility maintenance.


## Retention And Cleanup

- Move root-level ad-hoc scripts into `scripts/` or delete them. Remove `__pycache__` and `.DS_Store` before finishing.
- Keep summaries and workpads. Delete redundant actor checkpoints for dropped profiles. Inspect the registry before deleting large artifacts.

## Skill Distillation & Auto-Optimization

- After a complex task (5+ tool calls) or tricky fix, evaluate whether the
  approach deserves a new or updated skill. Patch stale skills immediately.
- Every 3–5 related tasks, inventory `.codex/skills/`, deduplicate overlaps,
  refresh front-matter, and commit. Delete skills unused for 30 days that have
  been superseded.

## Auto-Commit Discipline

Every task that changes files in the repo **must** end with:

1. `git add` of all relevant changes.
2. `git commit` with a descriptive message.
3. `git push` to sync evidence to the remote.

Skip push only for confirmed permission or network failures.

## Default Posture

- Prefer improving shared harness paths over one-off scripts.
- Prefer explicit interfaces and validation over hidden conventions.
- Fix the harness, not just the symptom, when a failure exposes a gap.
- Keep unrelated churn out of scope.

## Environment

This workspace uses **`qlib_venv`** (Python 3.11). All scripts in `scripts/`,
`src/soxl_quant/`, and `production/` run against it. Requirements are frozen in
`requirements/qlib.txt`.

## Red Lines

- Do not exfiltrate private data.
- Do not run destructive commands without asking.
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

Use git worktrees only for **short-lived experimental branches**.
- Location: `.worktrees/<branch-name>/`
- Delete immediately after merge or abandon
- Do not use for long-term development

## Execution Standard

For every substantial iteration:

1. ground in the current repo state
2. update or confirm the active workpad only when the task is not a
   notebook-contained experiment
3. run exactly one meaningful loop
4. validate the result
5. write proof of work back to the repo, preferably as or from a runtime
   notebook when the task is experimental or diagnostic
6. put single-run conclusions in the experiment notebook; update
   `runtime/<direction>/knowledge.md` only for durable cross-session decisions
7. **commit and push the changes**
8. if a skill was created or modified, verify `.codex/skills/` is synced

