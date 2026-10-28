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

The active Stage1 historical research Goal is defined in
`docs/stage1_historical_goal.md`. Its completion is a fully gated 0/2/5bp
historical Qlib account result, recorded as `historical_research_passed`;
independent forward evidence remains a separate contract-v2 promotion gate.

## Current research baseline and failure-to-feature loop

- Start new economic mechanisms from the frozen, complete Stage1 allocation,
  action, risk, and recovery chain. Reuse its matching Qlib source and native
  account as the attribution baseline; keep SOXX as the final benchmark. Do not
  silently switch the parent to a different allocator or a strong reference.
- The user separately authorized an independent XGB/LGB three-asset allocator
  comparison in `research/independent_tree_allocation_20261027`. It is an
  explicit separate route, not a matched f696+b472 receiver or a change to the
  active release. Its fixed 0bp native-account version failed the SOXX risk,
  year/window and/or wealth gates; retain the evidence and do not tune that
  version on the reused 2023-2026 history. Stage1's historical Goal definition
  remains unchanged.
- `stage1_nvda_guidance_receiver.prepare_account_sources` returns the raw
  cashbook frame containing **27baa** allocation proposals. It does not replace
  them with f696 outputs. For an f696-parent experiment, explicitly align and
  substitute all five saved `inputs/source_targets["frozen_current"]` outputs,
  verify 944/944 proposal rows against the frozen A_0 source, then run the
  candidate's own-state account. The identity audit is
  `research/stage1_f696_parent_identity_audit_20261023`.
- The saved 2018–2022 original-architecture physical training reference is also
  a different parent: its raw three-asset proposals disagree with f696
  `development_walk_forward` on all 1056 shared signal dates. Its fee-net local
  labels and fixed information failures retain their original scope; they are
  not matched f696 training labels or automatic necessary gates. The read-only
  audit is `research/stage1_f696_training_label_parent_identity_audit_20261025`.
  Before fitting a joint f696 receiver, build and verify an exact-parent physical
  0bp Tiered development reference and separately freeze its local labels.
- The covered exact-parent development reference is now
  `research/stage1_f696_exact_parent_training_reference_20261027`: 1056
  2018–2022 signals have f696 five-output parity and a native f696+b472 0bp
  Tiered own-state account. Earlier 700 f696 development signals lack matching
  09:46/09:47 minute history; five terminal physical-source dates lack f696
  proposals. The paired local-label card has 617 binding cash-to-SOXL and 304
  nonbinding beta-neutral SOXX rotation labels. The fixed state-plus-prior-market
  and 09:46 volume-timing information screens both failed their joint gates;
  neither authorizes a 2023+ candidate account. Preserve all five cards and
  stop those versions without sign, year, threshold, or feature-window rescue.
  The read-only follow-up is
  `research/stage1_f696_joint_failure_attribution_20261027`: 2022 binding
  predictions stayed positive while realized local value weakened, and the
  nonbinding route had only one eligible 2022 label. These are descriptive
  shifts, not a tradable VIX, drawdown, or cash threshold.
- In `qlib_venv`, set `MLFLOW_ALLOW_FILE_STORE=true` before importing Qlib
  workflow or reading the sealed local Recorder stores. Without it, MLflow may
  report `No valid experiment has been found` even when the experiment ID,
  directory and hashes are correct. The read-only runtime proof is
  `research/stage1_f696_native_source_runtime_preflight_20261023`. This
  fixes source loading only; it does not reopen a scientifically stopped plan.
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
- Read the exact f696 A_0 0bp cash-layer bound before another cash-only feature:
  `research/stage1_original_0bp_cash_layer_window_bound_20261023`. On the
  frozen path, 255/819 rolling windows beat SOXX; unrealistically erasing all
  negative cash contribution from cap-binding days alone reaches 321, and from
  nonbinding days alone reaches 420, below the required 574. Erasing both reaches
  588. This accounting exercise is not a tradable counterfactual or a proof that
  every one-layer policy fails. It means the next mechanism must address how
  allocation intent and own-state risk capacity jointly reach the funded action,
  then prove the result in its own complete 0bp account.
- Correcting a prior experiment's parent identity is an audit repair, not a new
  economic hypothesis. Do not automatically rerun a static composition, cap,
  bypass, or other old intervention on f696 merely because its earlier account
  used 27baa. First show which exact f696 failure interval and matched control
  motivate a distinct decision-time feature and receiver.
- Convert a repeatable failure mechanism into one falsifiable feature hypothesis.
  Check what Stage1 already receives and what closed experiments already tested;
  verify source identity, first availability, missingness, and label maturity.
  Freeze the receiver, control, budgets, and information gate before fitting.
  Judge any accepted feature on the complete native account and all regimes,
  including intervals outside the motivating failure. A failed fixed version
  stays closed. The detailed loop is `docs/research_failure_to_feature_sop.md`.
- For the frozen f696 allocation source, `timing.decision_at` is the next US
  session at 09:29 ET. The b472 opening action decision is later, at 09:46 ET.
  A same-day 09:45 completed minute bar cannot enter the unchanged f696
  allocator. To use it, freeze an explicit later-clock allocator or a b472
  action receiver with its own economic label and controls before fitting.
- For the active Stage1-versus-SOXX Goal, do not initiate or use news reports,
  press releases, article text, or scraped news-derived series as candidate
  features. Prefer the frozen Qlib/NAS market data and existing decision-state
  evidence when proposing the next bounded mechanism.
- As confirmed by the user, Yanxuan NAS has no additional
  CRSP, Sharadar, ORATS, or comparable historical dataset available for this
  Goal. Continue with the existing Qlib/NAS market data and saved Stage1
  evidence. Do not treat an unlocated commercial source as an active dependency
  or ask again for a vendor or NAS directory. Recheck the current cards before
  reviving a source: ADS, G.17 industrial production, CFTC NQ positioning, and
  SOXX issuer-share creation have already had fixed information versions closed.
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
