# SOXL Qlib Workspace Specification

This specification is the active architecture for new work. Qlib is the source
of truth for market-data alignment, research workflows, recorders, reports, and
analysis evidence. The current three-asset campaign's scientific gates and
roles are defined in `docs/three_asset_research_contract.md`; task sizing,
model choice, monitoring, and archival procedure are defined in
`docs/research_operations.md`. Historical instructions are retained in
`docs/archive/` and apply only to artifacts that explicitly reference them.

Default implementation agents use Luna/medium, monitors Luna/low, and scoped
reviewers Sol/high. Dispatch with an explicit model and short independent
context; escalate concrete failures as needed. Do not default routine work to
the owner's model or reasoning effort.

## Active campaign boundary

New campaign work uses SOXL, SOXX, and cash. SOXX is the final fee-adjusted
performance hard gate. Stage1 is a frozen comparison baseline; matching controls
and pre-registered references support attribution and remain diagnostic. Do not
rewrite old cards or retroactively change their gates. The active campaign
projection is `research/three_asset_campaign.json`; detailed acceptance,
statistical, risk, cost, forward-evidence, and Goal lifecycle rules are in the
versioned contract.

The workspace supports research and paper/simulated execution. Do not buy data,
place real broker orders, or replace an active release through research code.
Missing evidence for one branch does not stop eligible work on another branch.

## Architecture and data authority

- `data/` holds canonical inputs, Qlib stores, snapshots, manifests, and source
  lineage. Record source identity, event/availability time, label maturity,
  calendar, provider, and snapshot/version for each result.
- `src/core/` defines schemas, manifests, hashes, configuration, and validation.
  `src/data/` builds and audits Qlib providers and snapshots. `src/factor/`,
  `src/model/`, and `src/objective/` implement Qlib Dataset-based factors,
  registered models, and classification/regression/ranking/risk-return objectives.
- `src/backtest/` owns Qlib signal/portfolio workflows and costs;
  `src/analysis/` wraps Recorder, report, and risk-analysis outputs into stable
  reports. Dashboard, schedules, and adapters consume registered manifests or
  exports from `src/`; they must not mine ad hoc notebook internals.
- `research/<experiment>/` contains research notebooks and slim card sidecars.
  `production/<strategy>/` contains promoted paper/sim notebooks, fixed inputs,
  and their recorder evidence. `tests/` covers the relevant shared contracts.
- Existing `soxl_backtest/` is compatibility/reference material. New reusable
  code belongs in `src/`; preserve legacy import paths and frozen evidence when
  referenced. Skills describe methodology only and are not executable entrypoints
  or storage for business logic, project state, or fixed parameters.

## Native Qlib requirements

Use Qlib APIs as documented at `https://qlib.readthedocs.io/en/latest/` for
calendar, instrument and feature access, Dataset construction, Recorder,
Workflow, backtest, reports, and risk analysis. Read the local experiment skill
before substantial Qlib research and consult the official docs before designing
or changing these API surfaces. If docs are unreachable, say so and do not guess.

Workflows must use `qlib.init`, `qlib.data.D` and/or Qlib Dataset, and native
`qlib.workflow.R` Recorder/Records where applicable. Do not treat raw parquet,
calendar files, pandas-only results, hand-built recorder IDs, or Qlib-named
folders as native Qlib evidence. Do not invent API behavior or silently fall
back to pandas when Qlib inputs are missing.

## Notebook and card contract

Research begins in `research/<experiment>/<experiment>.ipynb`; promoted runnable
strategies begin in `production/<strategy>/<strategy>.ipynb`. Each notebook
records its hypothesis or strategy, source/provider and timing, Qlib loading,
fit/predict/backtest steps, analysis, sidecars, and a final Chinese conclusion
covering work performed, metrics, regime/window differences, promotion decision,
risks, next action, and Recorder identity. Docs, tests, and small framework work
may be notebook-free.

Each card root remains small: notebook, `workflow.yaml`, `summary.json`,
`conclusion.md`, selected-model metadata, and compact analysis/backtest/risk
summaries. `summary.json` includes `artifact_policy.policy` set to
`slim_card_recorder_first`, workflow configuration, and the recorder-only
artifact list. Predictions, signals, all-model prediction matrices, portfolio
reports, and positions live in the Qlib Recorder. A requested temporary export
must be removed before commit. Keep sidecars synchronized with the notebook
conclusion. Validated research and production cards include a direct benchmark
comparison and explicit promotion decision; diagnostic search cards may fail.

## Artifact, compatibility, and execution constraints

Before generating artifacts, assign them to `data/`, `src/`, a research or
production card, or an external scratch location. Commit only durable evidence
needed to reproduce or audit a result. Exclude caches, logs, `__pycache__`,
`.DS_Store`, local locks, downloads, screenshots, and unregistered large model
files. Force-add only exact verified sidecars. Preserve required MLflow recorder
directories and record a real metric when needed. Never rebuild a repo-wide
index from a partial tree.

Do not recreate reinforcement-learning reward flows. Use objective contracts.
V1 broker-facing code is limited to adapter contracts and paper/simulated target
positions; no real order execution. Existing code with frozen references must
remain recoverable. Before cleanup, inspect imports, notebooks, registries,
frozen manifests and serialized artifacts; archive exact bytes, identity hashes,
source commit, and restoration path before deletion. If deletion safety is
uncertain, preserve the code and report that it remains.

## Verification and delivery

Use the repository Qlib environment and the smallest relevant existing check.
Run focused checks once after the final code change. Add tests only for a new
material contract or a concrete regression not caught by existing coverage.
Do not run broad suites unless a shared/cross-layer change or explicit request
justifies them. Documentation-only work requires diff review and
`git diff --check`, not pytest or notebook execution. Research still requires
the relevant notebook, native Qlib evidence, source/timing/cost checks, and
scientific gates; software tests do not substitute for them. Reuse verified
unchanged Recorder evidence and perform one targeted readback if delivery moves
the evidence layout.

Every change must preserve unrelated WIP, be staged by exact path, and receive
the appropriate verification, commit, and push unless blocked by network or
permissions. If a worktree was necessary, complete validation and delivery,
merge into the integration target, synchronize the primary checkout, and remove
only the clean merged worktree and branch before handoff.
