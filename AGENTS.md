# AGENTS.md

This repository is a `SOXL` harness workspace.

The job of the agent is to execute inside a repeatable research and validation
harness, leave evidence in the repository, and improve the harness when gaps are
exposed.

Default artifact posture: future SOXL experiments, diagnostics, factor screens,
backtests, parameter screens, model-selection loops, and one-off analytical
follow-ups should go to executable runtime notebooks first. Prefer
`runtime/<direction>/card/<experiment>/<experiment>.ipynb` as the primary
workflow and conclusion unless the task is pure code maintenance, production
operations, or explicitly asks for another format.

Standing notebook-first rule: for later model, factor, reward, network,
multi-asset, backtest, and broker-data analysis work, put the runnable workflow,
special-case logic, metrics, and single-run conclusion in the corresponding
runtime `.ipynb` by default. This preference overrides older habits of writing
single-run conclusions into `docs/active_workpad.md`, Markdown reports, or
knowledge files unless the result is promoted, cross-session coordination state,
or the user explicitly asks for those surfaces.

When a future task asks to record, summarize, compare, conclude, screen,
diagnose, or continue research work, treat "put it in ipynb" as the default
instruction even if the prompt is terse. Keep chat as a pointer to the notebook
and sidecar artifacts; do not create Markdown-only conclusions or update
`docs/active_workpad.md` unless the result is durable cross-session state,
production coordination, or explicitly requested.

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
- `runtime/<direction>/card/<experiment>/*.ipynb`: default executable
  workflow, run-local scratchpad, evidence, and primary single-run conclusion
  for experiments, diagnostics, backtests, factor screens, and model-selection
  loops.
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

If you want future sessions to remember something important from a single run,
put it in the experiment's runtime notebook first. Promote it into
`runtime/<direction>/knowledge.md` only when it becomes durable cross-session
context; use `notes.md` only when a profile/package explicitly requires it.

Knowledge retrieval must be scoped by task direction.

## Workpad Discipline

- Reuse the existing active workpad for cross-session coordination instead of
  scattering durable task state across random notes; do not use it as the
  default scratchpad or conclusion surface for notebook-contained work.
- Keep the workpad current only for active cross-session coordination,
  production operations, or durable task-state changes. If an experiment,
  diagnostic, backtest, parameter screen, or model-selection loop is fully
  contained in a runtime notebook, leave `docs/active_workpad.md` unchanged by
  default.
- Do not add single-run experiment conclusions to the workpad by default. Put
  the executable workflow, special-case logic, metrics, and conclusion in the
  corresponding runtime notebook instead.
- Capture baseline or reproduction evidence before changing code when the task
  depends on a behavior change.
- A task is not complete until the repository contains the proof of work.

## Notebook Experiment Discipline

- Notebook-first is the standing default for future SOXL research work and
  follow-up turns. Treat the user's preference as: "尽量都到 ipynb" unless the
  task is purely code maintenance, operational startup, or explicitly asks for
  another artifact.
- Treat this as durable repo policy for future sessions, not a one-turn
  preference. When there is uncertainty, choose the runtime notebook path first
  and keep `docs/active_workpad.md` unchanged unless coordination state truly
  needs to move there.
- For future work, start experiment and diagnostic execution in an executable
  runtime notebook by default. Do not treat Markdown notes, chat conclusions,
  sidecar JSON, or one-off scripts as the primary result when an `.ipynb` can
  hold the runnable workflow and conclusion.
- This default applies even when the user gives a terse continuation prompt.
  Start from the relevant runtime notebook surface unless the task clearly
  requires a shared code or production change first.
- For continuation work, first look for an existing runtime notebook for the
  active direction and update it when appropriate; create a new notebook only
  when the follow-up is a distinct experiment, diagnostic, or backtest.
- For follow-up prompts such as `继续`, `开始`, `试试`, `其他看看`, or
  `continue`, find or create the relevant runtime notebook before adding
  Markdown notes, workpad entries, or knowledge-file updates.
- Do not wait for the user to repeat this preference. Experiments, diagnostics,
  factor screens, model/reward/network trials, broker-data audits, backtests,
  parameter screens, and one-off analyses should put the executable workflow and
  conclusion in a runtime notebook by default.
- If a non-notebook primary artifact is chosen for experiment or diagnostic
  work, state the reason in the final answer or the relevant durable note.
- Default to an executable notebook under
  `runtime/<direction>/card/<experiment>/<experiment>.ipynb` for experiments,
  diagnostics, parameter screens, backtests, and one-off research loops.
- Put single-run special-case logic, metrics, comparisons, conclusions, and
  artifact links in that notebook. The notebook should be runnable evidence, not
  just a narrative summary.
- The runtime notebook is the primary human-readable conclusion for a single
  run. Sidecar Markdown/JSON/CSV files may support indexing, machine-readable
  selection, or summaries, but they must not replace the notebook as the source
  of the run's conclusion.
- Every executed experiment or diagnostic notebook should end with a concise
  conclusion cell that records what was run, key metrics, promotion/drop
  decision, risk notes, artifact links, and the next action. Do not leave the
  single-run conclusion only in chat, a workpad, a Markdown report, or sidecar
  JSON.
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
- For future experiment or diagnostic follow-ups, create or update the relevant
  runtime notebook first. Promote conclusions into workpads, knowledge files,
  Markdown reports, or research knowledge only after they become durable project
  state or the user explicitly requests that form.
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

- `src/soxl_quant/app/` for orchestration, `core/` for config/helpers, and
  runtime notebooks for single-run experiment evidence.
- `src/soxl_quant/settings.py` owns repo paths, constants, and artifact naming only.
- `src/soxl_quant/config.py` owns typed config dataclasses and profile loading.
- Do not recreate retired facades `src/soxl_quant/cli.py`,
  `src/soxl_quant/rl.py`, or the legacy trainer entrypoint
  `src/soxl_quant/app/cli.py`.
- Legacy RL helper logic that still exists under `src/soxl_quant/policy/` must
  remain isolated from the default supervised entrypoint until it is migrated or
  deleted.


## Script Boundaries

- New or modified files under `scripts/` should import from `src/soxl_quant/app/`, `src/soxl_quant/policy/`, `src/soxl_quant/core/`, or `src/soxl_quant/runtime.py` directly.
- Do not point new script work at retired facades `src/soxl_quant/rl.py`,
  `src/soxl_quant/cli.py`, or `src/soxl_quant/app/cli.py`; those paths should
  stay deleted.


## Directory Read Order

When orienting to the repo, prefer indexes and summaries before raw outputs.

- `experiments/`: index → README → profile.yaml → expectations → notes
- `docs/research_knowledge/`: index → task_router.json → relevant mechanism file
- `runtime/`: README → index.json → summaries before raw logs/CSV
- `runtime/<direction>/knowledge.md`: read after choosing a direction, then open the corresponding paths

## Compatibility Policy

- Retired compatibility facades:
  - `src/soxl_quant/cli.py`
  - `src/soxl_quant/rl.py`
- Retired legacy trainer entrypoints:
  - `src/soxl_quant/app/cli.py`
  - `src/soxl_quant/app/train.py`
  - `src/soxl_quant/app/workers.py`
  - `src/soxl_quant/app/selection.py`
- Retired RL policy modules:
  - `src/soxl_quant/policy/reward.py`
  - `src/soxl_quant/policy/environment.py`
  - `src/soxl_quant/policy/evaluation.py`
  - `src/soxl_quant/policy/policy_builder.py`
  - `src/soxl_quant/policy/models/attention.py`
  - `src/soxl_quant/policy/models/gru.py`
  - `src/soxl_quant/policy/models/mlp.py`
  - `src/soxl_quant/policy/models/moe.py`
  - `src/soxl_quant/policy/models/temporal_conv.py`
  - `src/soxl_quant/policy/models/tide.py`
- Retired RL compatibility/bootstrap modules:
  - `src/soxl_quant/compat.py`
  - `src/soxl_quant/app/bootstrap.py`
- Retired RL environment tests:
  - `tests/test_structural_model.py`
- Retired BC/actor-builder harness tests:
  - `tests/test_budget_usage_bc.py`
  - `tests/test_state_objective_bc.py`
  - `tests/test_policy_networks.py`
  - `src/soxl_quant/app/state_objective_bc.py`
- Do not recreate retired facades or trainer entrypoints. Default package entrypoints should
  continue through the simplified supervised architecture CLI unless the user
  explicitly asks for legacy compatibility maintenance.
- Legacy actor runtime wrappers under `production/` are retired. The daily
  paper runner uses the supervised Alpha158/cadence runtime artifact and must
  fail closed until a promoted supervised daily signal path exists.


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
- single-run experiment workflows and conclusions:
  `runtime/<direction>/card/<experiment>/<experiment>.ipynb`
- workpads and cross-session decision trail: `docs/active_workpad.md`
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
5. write proof of work back to the repo as a runtime notebook first when the
   task is experimental, diagnostic, analytical, or model-selection related
6. put single-run conclusions in the experiment notebook; update
   `runtime/<direction>/knowledge.md` only for durable cross-session decisions
7. **commit and push the changes**
8. if a skill was created or modified, verify `.codex/skills/` is synced
