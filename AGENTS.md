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

At the beginning of every new task, including follow-up tasks:

1. Run `git status --short` and `git worktree list --porcelain`.
2. Identify the integration target (default `main`), unfinished worktrees, and
   leftover local/remote task branches; inspect `git branch --all` as well.
   Summarize large existing WIP instead of repeatedly printing unchanged paths.
3. Use the primary checkout by default and preserve unrelated WIP. Check whether
   an existing worktree can be completed or reused before considering another.

Before substantial work:

1. Read `AGENTS.md`.
2. Read `SPEC.md`.
3. Satisfy the qlib docs and skill requirements above.
4. Pick the relevant notebook surface:
   - `research/<experiment>/<experiment>.ipynb` for research work.
   - `production/<strategy>/<strategy>.ipynb` for promoted runnable strategies.

## Worktree Use And Closure

- Avoid worktrees by default. Read-only analysis, documentation, narrow fixes,
  and evidence bookkeeping should use the primary checkout when unrelated WIP
  can be preserved with exact file staging.
- Create a worktree only when isolation is necessary for conflicting concurrent
  work or incompatible changes. State that concrete reason briefly; do not
  create one merely as a standard startup step.
- At task startup, inspect leftover worktrees for dirty state, unique commits,
  and ancestry to the integration target. Complete and merge useful in-scope
  work; remove only clean worktrees whose changes are already merged.
  Preserve dirty or unmerged work that belongs to another task and report it.
- If this task uses a worktree, finish validation, commit, push, merge into the
  integration target, synchronize the primary checkout, and remove the clean
  merged worktree and its task branch before handoff. Do not leave delivery or
  cleanup to a later task.
- After cleanup, check `git worktree list --porcelain` once. Every subsequent
  new task must check again rather than assume the previous state still holds.
- Include leftover task branches in startup and closure checks even when no
  extra worktree is registered. Delete completed branches already merged into
  the integration target; preserve active branches with unique unmerged changes.
- When cleanup of obsolete branches is requested, retain their exact tips in
  remote archive tags and verify those tags before deleting branch refs. Check
  for active worktrees and open PRs first; do not merge obsolete implementations
  into `main` merely to remove their branches.
- Never reset, discard, force-remove, or stash unrelated WIP to simplify this
  process. Do not rebuild repository-wide artifact indexes from a partial tree.

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
- Git-delivered MLflow file Recorders must retain their required directories.
  Persist a real run metric (such as registered artifact count) when the metrics
  directory would otherwise be empty; do not use an invalid metric placeholder.

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

Use the repo qlib environment and the smallest meaningful verification for the
changed layer. Prefer existing tests; do not expand the test surface by default.

- Add a test only for a new material contract or a concrete regression that
  existing coverage cannot detect. Do not write tests for documentation,
  routine artifact bookkeeping, low-impact reversible edits, or assertions
  that merely mirror the implementation. Avoid one testcase per unchanged
  field, duplicate fixtures, and repeated coverage of the same behavior.
- Run the relevant focused checks once after the final code change. Repeat or
  broaden them only for a subsequent code change, a new failure, or an unresolved
  material concern. Commit, push, and merge alone do not justify another test run.
- Run broad qlib suites or full pytest only for changes to shared contracts,
  cross-layer behavior, or an explicit request. Record known baseline failures
  once; do not keep rerunning or fixing unrelated archived/sparse-checkout tests
  while delivering a scoped experiment.
- Research still requires the relevant notebook execution, native Qlib evidence,
  source/timing/cost checks, and scientific gates. These requirements are not
  replaced by fewer software tests. Reuse verified unchanged Recorder evidence;
  do not refit models or replay backtests just for prose, file copying, or Git
  delivery. Use one targeted readback when delivery changes the evidence layout.
- Documentation-only work normally needs diff review and `git diff --check`;
  it does not need pytest, notebook execution, or a new worktree.

Choose the applicable commands; these are not an every-task checklist:

```bash
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/<relevant_existing_test>.py -q
PYTHONPATH=. qlib_venv/bin/python -m pytest tests/test_qlib_native_workflow.py tests/test_qlib_model_selection_validation.py tests/test_qlib_native_notebook_cards.py -q
PYTHONPATH=. qlib_venv/bin/python -m pytest -q
PYTHONPATH=. qlib_venv/bin/python -m nbconvert --execute research/<experiment>/<experiment>.ipynb --to notebook --inplace
```

Every file-changing task should finish with the appropriate minimal verification,
commit, and push unless network or permission failures prevent it.

## Negative Research Results

### Continuous Goals And Independent Evaluation

- For an explicitly authorized continuous research Goal, closing a failed fixed
  experiment closes that version, not the overall objective. Continue through
  failure attribution, winning/losing interval comparison, and a new falsifiable
  mechanism within the authorized campaign. Do not repeat a closed experiment
  without a substantive new mechanism or evidence condition.
- The research owner maintains the campaign state and next action; the research
  executor implements experiments; an independent evaluator checks the frozen
  design and evidence. The executor cannot certify its own promotion. Evaluation
  rejects candidates, not the research objective, and must identify concrete
  failed gates or the minimum missing evidence rather than request broad audits.
- After a review handoff, check the executor's actual agent/job status and
  explicitly resume an idle/completed executor. A queued message, an accepted
  plan, or a run-state file does not prove that work is running. Poll an existing
  live handle rather than restarting work after an observation timeout.
- Freeze experiment scope, sources, cost scenarios, selection rules and gates
  before execution. Development feedback may guide another experiment; any
  evaluation data used for that feedback is no longer independent validation.
  A separate reviewer or agent does not make reused market data independent.
- A single data-source blocker does not block all research. Maintain explicit
  branch dependencies; use the actual Goal lifecycle rules for overall status.
  Never mark a Goal complete because a notebook, test suite, or delivery finished.
- The active three-asset campaign contract is
  `docs/three_asset_research_contract.md`; its current state is
  `research/three_asset_campaign.json`. These do not rewrite sealed prior cards.

- Once a fixed experiment returns `not_promoted`, synchronize its conclusion
  and retain the minimum auditable evidence, then complete only the necessary
  delivery checks and stop that route.
- Do not turn a negative result into more testcase writing, repeated full-suite
  runs, extra audit inventories, or another parameter search. Additional research
  must belong to an already authorized bounded campaign or a new user request.
- Report the failed scientific gate and the next concrete information need.
  A rejected experiment is a completed experiment, not an achieved stable-excess
  objective; software validation must not become an open-ended substitute.
