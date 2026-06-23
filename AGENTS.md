# AGENTS.md

This repository is being rebuilt as the `SOXL` qlib-managed research and
paper-execution workspace. Qlib is the system of record for data alignment,
workflow evidence, analysis artifacts, and notebook-backed conclusions.

The current rebuild direction is defined in `SPEC.md`. Do not use chat history
as durable state.

Default artifact posture: qlib research and production work is
notebook-first, recorder-first, and slim-card by default. The card root is the
human and machine entry surface; large predictions, signal outputs, portfolio
reports, positions, and all-model prediction matrices live in the qlib recorder.

## Qlib Docs And Skill Requirements

These requirements are mandatory for all substantial work in this repository:

- Read `.codex/skills/soxl-qlib-experiment/SKILL.md` before planning or editing
  qlib-backed research, factor, model-selection, backtest, analysis, or notebook
  work.
- Use the official qlib documentation as the authority for qlib APIs and
  workflow behavior: `https://qlib.readthedocs.io/en/latest/`.
- Check the official qlib docs before changing or designing code that touches
  qlib data layout, calendars, instruments, Dataset, Recorder, Workflow,
  Analysis, Backtest, reports, or risk analysis.
- If the local skill and official qlib docs conflict, follow the official qlib
  docs for API behavior and update the local implementation plan accordingly.
- If the official qlib docs are unreachable, say so explicitly and do not invent
  qlib API behavior.

## Startup

Before substantial work:

1. Read `AGENTS.md`.
2. Read `SPEC.md`.
3. Satisfy the qlib docs and skill requirements above.
4. Pick the relevant notebook surface:
   - `research/<experiment>/<experiment>.ipynb` for research work.
   - `production/<strategy>/<strategy>.ipynb` for promoted runnable strategies.

## Active Surfaces

- `data/`: canonical local data, qlib calendar/store files, raw inputs, and
  reproducible snapshots.
- `src/`: reusable project modules for core contracts, data, factors, models,
  objectives, backtesting, analysis, dashboard exports, schedules, and adapters.
- `research/<experiment>/`: research notebooks, slim card-local sidecars, and
  qlib recorder evidence.
- `production/<strategy>/`: promoted strategy notebooks, slim card-local
  sidecars, fixed strategy inputs, paper/sim outputs, and qlib recorder
  evidence.
- `tests/`: contract, data, factor, objective, model, backtest, analysis,
  dashboard/export, schedule, and paper-adapter tests.
- `.codex/skills/`: methodology and workflow guidance only.

Existing `soxl_backtest/` code is compatibility/reference material during the
rebuild. New reusable functionality should go under `src/` unless a failing test
proves a narrow compatibility fix is needed.

## Qlib-First Rule

- Use qlib-managed calendars, stores, recorders, reports, and analyzers wherever
  the task is about data loading, training evidence, backtest evidence, or
  performance analysis.
- Keep data lineage explicit: every notebook result should identify its qlib
  calendar/store or canonical snapshot source.
- Analysis code should wrap qlib recorder/analyzer outputs into stable project
  reports instead of parsing scattered experiment files directly.
- Dashboard, promotion, and schedule surfaces should read manifests/exports from
  `src/` outputs, not ad hoc notebook internals.

## Notebook Rule

Research and production work starts in an executable notebook unless the task is
only a small framework/test/documentation change.

Each notebook should contain the hypothesis or strategy, qlib-backed data
loading, model/factor/backtest steps, analysis, sidecar links, and a final
Chinese conclusion cell covering:

- what ran
- key metrics
- regime/window differences
- promotion decision
- risks
- next action

CSV and JSON sidecars support the notebook. They do not replace the conclusion.

## Qlib Card Contract

Every qlib research or production card must keep these root files small and
reviewable:

- `<experiment>.ipynb` or `<strategy>.ipynb`
- `workflow.yaml`
- `summary.json`
- `conclusion.md`
- selected-model metadata and small model-selection/analysis/backtest/risk
  summaries such as `selected_model.json`, `model_selection.csv`,
  `risk_analysis.csv`, `backtest_benchmark_comparison.csv`,
  `analysis_summary.json`, `regime_metrics.csv`, and `window_metrics.csv`

The following artifacts are recorder-only unless a task explicitly asks for a
temporary local export and that export is removed before commit:

- `predictions.csv`
- `signals.csv`
- `all_model_predictions.csv`
- `portfolio_report.csv`
- `qlib_portfolio_report.csv`
- `qlib_positions.csv`

`summary.json` must include `artifact_policy.policy:
slim_card_recorder_first`, `artifact_policy.workflow_config`, and
`artifact_policy.recorder_only_artifacts` for qlib cards.

Search cards may be diagnostic and may record a candidate that fails the legacy
or benchmark comparison. Validated research cards and production cards are the
promotion gates; they must record the direct benchmark comparison and state the
promotion decision in the notebook conclusion.

## Artifact Control

Generated artifacts must be controlled, reviewable, and tied to a notebook,
manifest, or data snapshot.

- Before generating artifacts, decide where they belong:
  - `research/<experiment>/` for research notebook sidecars.
  - `production/<strategy>/` for promoted strategy sidecars.
  - `data/` for canonical snapshots and qlib stores.
  - outside the repo for temporary SDK logs, caches, scratch downloads, and
    one-off debug output.
- Commit only durable artifacts needed to reproduce or audit a result: executed
  notebooks, `summary.json`, `conclusion.md`, small CSV/JSON sidecars,
  manifests, and schema/version metadata.
- Do not commit transient caches, SDK logs, `__pycache__`, `.DS_Store`, local
  lock files, temporary downloads, ad hoc screenshots, or large model/checkpoint
  binaries unless the task explicitly requires them and the artifact has a
  manifest entry explaining source, date range, and purpose.
- If `.gitignore` hides an intentional sidecar, use `git add -f` only for the
  exact verified files. Do not force-add broad globs.
- Keep notebook outputs and sidecars synchronized: rerun or update the notebook
  conclusion after changing generated CSV/JSON metrics.
- Rebuild runtime or artifact indexes only in a workspace that has the complete
  expected artifact set. Do not treat an index rebuilt from a partial worktree as
  repo-wide truth.
- Remove obsolete rollback backups, RL-era checkpoints, local training caches,
  scheduler logs, root screenshots, and root one-off result text files when they
  are not referenced by a qlib notebook, `summary.json`, `workflow.yaml`,
  manifest, or registry row.
- Before every commit, run `git status --short` and inspect the staged diff so
  unrelated generated files are not included.

## Skill Boundary

Skills must not implement business logic, store project state, hold fixed
parameters, or become functional entrypoints.

Use skills for:

- research method prompts
- hypothesis decomposition
- factor/regime/model-difference thinking
- reminders about qlib workflow shape

Move concrete paths, results, parameters, manifests, and project state into
`src/`, `research/`, `production/`, or `data/`.

## Rebuild Constraints

- Do not recreate reinforcement-learning reward flows. Use `objective` concepts
  for classification, MSE, IC, ranking, and risk/return utility.
- Do not implement real broker execution in v1. Broker-facing work is limited to
  adapter contracts and paper/simulated target-position flows.
- Do not put production dashboard, scheduler, or broker behavior in skills.
- Do not add scattered scripts as primary workflows. Prefer `src/` modules plus
  executable notebooks.

## Verification

Use the repo qlib environment for validation:

```bash
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_qlib_native_workflow.py tests/test_qlib_model_selection_validation.py tests/test_qlib_native_notebook_cards.py -q
PYTHONPATH=. qlib_venv/bin/python -m pytest -q
PYTHONPATH=. qlib_venv/bin/python -m nbconvert --execute research/<experiment>/<experiment>.ipynb --to notebook --inplace
```

For new rebuild work, add focused verification that matches the changed layer.
Every file-changing task should finish with focused verification, commit, and
push unless network or permission failures prevent it.
