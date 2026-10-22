# AGENTS.md

This repository is the SOXL Qlib-managed research and paper-execution workspace.
Qlib is authoritative for data alignment, workflow evidence, analysis artifacts,
and notebook-backed conclusions. `SPEC.md` defines the active architecture;
`docs/three_asset_research_contract.md` defines the current campaign gates;
`docs/research_operations.md` defines bounded tasks, model routing, monitoring,
and archival practice. Do not use chat history as durable project state.

The current campaign is version 2: SOXX is the final performance hard gate;
other comparisons support attribution and remain diagnostic. Frozen historical
cards keep their original gates. The active campaign entry is
`research/three_asset_campaign.json`.

## Current research baseline and failure-to-feature loop

- Start new economic mechanisms from the frozen, complete Stage1 allocation,
  action, risk, and recovery chain. Reuse its matching Qlib source and native
  account as the attribution baseline; keep SOXX as the final benchmark. Do not
  silently switch the parent to a different allocator or a strong reference.
- Screen a Stage1-derived candidate at 0bp added slippage first, with frozen
  IBKR Tiered fees still charged and its own full funded account. Report SOXX
  wealth, available-mark drawdown, CVaR5, complete years, all contiguous 126-day
  windows, and the frozen uncertainty test. A 0bp pass only advances research;
  contract v2 still requires separate 2/5bp paths and qualified forward evidence.
- Diagnose the entire matched Stage1-versus-SOXX history before freezing a
  feature experiment: retain every complete year, quarter, 126-day window,
  and both partial endpoint years. Identify weak intervals and decision-time
  regimes, then trace prediction -> target -> risk projection -> action ->
  holdings -> fees/NAV.
  Separate cash opportunity cost, SOXL-relative selection, execution cost, and
  risk-state feedback. Compare bad intervals with good intervals and matched
  controls; do not choose a tradable switch or threshold from hindsight groups.
- Convert a repeatable failure mechanism into one falsifiable feature hypothesis.
  Check what Stage1 already receives and what closed experiments already tested;
  verify source identity, first availability, missingness, and label maturity.
  Freeze the receiver, control, budgets, and information gate before fitting.
  Judge any accepted feature on the complete native account and all regimes,
  including intervals outside the motivating failure. A failed fixed version
  stays closed. The detailed loop is `docs/research_failure_to_feature_sop.md`.
- Use Yanxuan NAS as the durable backup for each campaign market snapshot,
  received original-source bytes, and large Qlib Recorder evidence. For the
  `us_semis_intraday` feed it is also the current source authority, as declared
  in `data/nas_sources.json`.
  Before using a source or treating a local cache as disposable, check its
  manifest, SHA-256, size, NAS receipt, and scoped restore/readback. A per-card
  archive does not prove every workspace file is backed up. Preserve any local
  object whose NAS copy or restoration has not been verified; report the gap.

Default new implementation agents to Luna/medium, monitors to Luna/low, and
scoped reviewers to Sol/high. Use short independent context and request the
model explicitly. Escalate concrete failures instead of inheriting the owner's
full conversation or highest reasoning effort.

## Start and close work safely

- At every task start, run `git status --short`, `git worktree list --porcelain`,
  and `git branch --all`. Default integration target is `main` and default
  checkout is the primary checkout. Identify dirty or unfinished worktrees and
  leftover task branches. Summarize large WIP; preserve it.
- Before substantial work, read this file and `SPEC.md`. For Qlib-backed
  research, factors, models, backtests, analysis, or notebooks, also read
  `.codex/skills/soxl-qlib-experiment/SKILL.md` and the official docs at
  `https://qlib.readthedocs.io/en/latest/`. The official docs govern API behavior.
- Use a worktree only for actual isolation needs. Do not reset, stash, clean,
  force-remove, or overwrite unrelated WIP. Archive exact source bytes and
  record hashes before deleting obsolete material; retain referenced or
  uncertain artifacts. Do not rebuild repo-wide indexes from partial trees.
- Before a commit, inspect `git status --short` and the staged diff; stage exact
  files. Complete the appropriate focused verification, commit and push unless
  blocked by network or permissions. If a worktree is used, merge to the
  integration target, sync the primary checkout, and remove only clean merged
  worktrees and branches before handoff.

## Qlib research and artifacts

- Use native Qlib calendars, stores, `D`, Dataset, Recorder, Workflow, reports,
  and analyzers for their respective data, training, backtest, and analysis
  tasks. Preserve source lineage, availability timing, and snapshot/provider
  identity. Never substitute pandas-only calculations or hand-built recorder
  IDs for a Qlib workflow. New reusable code belongs under `src/`; legacy
  `soxl_backtest/` is compatibility/reference material.
- Research starts in `research/<experiment>/<experiment>.ipynb`; promoted
  paper/sim strategies start in `production/<strategy>/<strategy>.ipynb`.
  Each notebook documents hypothesis/strategy, Qlib data, method, analysis,
  sidecars, and a Chinese conclusion with run, metrics, regime/window results,
  promotion decision, risks, and next action. Small framework, test, and docs
  changes do not require a research notebook.
- Keep card roots slim and recorder-first: notebook, `workflow.yaml`,
  `summary.json`, `conclusion.md`, selected-model metadata and small summaries.
  `summary.json` must declare `artifact_policy.policy:
  slim_card_recorder_first`, workflow config, and recorder-only artifacts.
  Predictions, signals, all-model matrices, portfolio reports, and positions
  belong in the Qlib recorder unless a temporary export is explicitly requested
  and removed before commit. Keep notebook conclusions and sidecars synchronized.
- Store canonical data and snapshots under `data/`, reusable modules under
  `src/`, research/production evidence with its card, and temporary caches,
  logs, downloads, and screenshots outside Git. Commit only durable, auditable
  evidence. Preserve required MLflow recorder directories and real metrics.
- Skills describe methodology only. They must not hold business logic, fixed
  project parameters/state, or functional entrypoints. Do not recreate RL
  reward flows; use objective concepts. V1 permits paper/sim adapters only and
  forbids real broker execution.

## Verification and scientific decisions

- Run the smallest relevant existing check once after the final code change.
  Add tests only for a material new contract or regression existing coverage
  misses. Documentation-only changes need diff review and `git diff --check`,
  not pytest or notebook execution. Broad suites are for shared/cross-layer
  changes or explicit requests.
- A research result requires the relevant executable notebook, native Qlib
  evidence, timing/source/cost checks, and frozen scientific gates. Do not
  retrain or replay for prose or Git delivery. Reuse verified evidence and do
  one targeted readback when its layout changes.
- Freeze scope, sources, costs, selection rules, budgets, and gates before
  execution. Development feedback may guide another experiment; reused market
  data cannot become independent validation because of a new card or reviewer.
  Executors cannot certify promotion; independent reviewers report concrete
  failed gates or the minimum missing evidence and do not expand into unrelated
  audits.
- A fixed `not_promoted` route closes after necessary attribution and minimum
  auditable evidence. Do not resume it through threshold/window tuning on reused
  history, redundant tests, or repeated inventories. Further work must use an
  authorized bounded campaign or a new user request.
- For an explicitly authorized continuous Goal, a failed fixed experiment closes
  that version, not the overall objective. Continue through failure attribution,
  interval comparison, and a new falsifiable mechanism within the campaign.
  Keep dependencies explicit; one missing source does not block unrelated work.
  A Goal is complete only when its objective is achieved, and may be paused only
  at the user's explicit request. Mark it blocked only after the same blocker
  recurs for at least three consecutive Goal turns and progress needs external
  change or user input. After review handoff, verify actual executor/job status;
  resume an idle executor explicitly and poll a live handle rather than restart
  it after an observation timeout. A queued message or state file is not proof
  of execution. Never treat software completion as scientific success.

Detailed current rules live in `docs/three_asset_research_contract.md` and
`docs/research_operations.md`. Historical contracts and superseded instructions
are kept under `docs/archive/`; do not apply them to new work unless a frozen
card explicitly references them.
