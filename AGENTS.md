# AGENTS.md — SOXL / SOXX / CASH research

## One research objective

Build a long-only, unlevered allocation among **SOXL, SOXX, and cash** that
earns stable, fee-net excess over SOXX. These are the only portfolio assets.
Other already held Qlib/NAS series may be factor inputs, never additional
holdings. Use the current data inventory; do not make a new vendor, purchase,
download campaign, or dataset expansion the default next step. Papers, official
documentation, and external analysis may inspire hypotheses and methods, but
are not historical observations or proof of a usable point-in-time feature.
Do not use news/article text as a candidate feature or invent missing history.

The historical Goal passes only when one preregistered three-asset candidate
passes the full **SOXX** 0/2/5bp native Qlib own-account gates in
`docs/three_asset_research_contract.md` and `docs/stage1_historical_goal_brief.md`:
same-basis net wealth, available-mark drawdown, CVaR5, complete-year and
rolling-126 stability, and block uncertainty. Zero added slippage still pays
IBKR Pro Tiered fees. A source, factor IC, model score, local action value,
Stage1 improvement, or software test is not historical SOXX success. Record
`historical_research_passed` only after all historical gates and independent
review. Independent forward evidence and production promotion are separate.
Research never places real orders or changes the active paper SIM release.

`SPEC.md` defines architecture; the research contract defines gates;
`docs/research_failure_to_feature_sop.md` defines the bounded hypothesis
funnel; `docs/stage1_factor_retention_policy_20261101.md` defines factor
retention. `research/three_asset_campaign.json` is the short active entry.
Card notebooks, frozen plans, Qlib Recorders, and their receipts hold detailed
history. Do not use this instruction file or chat history as an experiment log.

## Research loop using existing data

1. Start from a verified SOXX-relative account error. Trace factor or model
   output → proposed target → risk projection → legal whole-share order →
   actual holdings, cash, dividends and fees → net NAV. Compare complete years,
   quarters, all rolling windows, weak intervals, and matched strong intervals.
   Hindsight intervals explain failure; they do not define a trading switch.
2. Propose a **causal economic mechanism**, including what is known at the
   decision clock, where it enters allocation or risk, how it may change a
   funded action, how it may hurt good intervals, and what would falsify it.
   Think through new single-factor and multi-factor interactions in existing
   data. Use external research as a lead, verify its claim at the primary
   source, and adapt it to this three-asset time-series/account task.
3. Search prior cards by formula, input, economic mechanism, receiver and
   decision clock, not only card name. A closed fixed version remains closed;
   changing its sign, window, threshold, model capacity, or name on reused
   history is not a new mechanism. A genuinely different interaction, target,
   state path or decision-time information can be assessed on its own frozen
   terms. A finite inventory failure is not proof all existing data are useless.
4. Separate raw factor information, conditional information beyond matched
   controls, prediction change, funded action change, and full-account value.
   Retain an informative single factor when one combination fails. Correlation
   is a redundancy clue: compare baseline, replacement and joint use on the
   same rows, folds, labels, policy and costs before deciding. Negative IC may
   be information, but choose its sign only in training data. Do not call an
   uncorrected search or many overlapping minute rows independent evidence.
5. Freeze one bounded experiment before fitting: source identity and first
   availability, action/label maturity, train and evaluation windows, full-day
   grouping and purge, formula and missingness, matched controls, own-state
   fees/risks, finite fit/account budget, and failure conditions. The SOP's
   default batch is at most eight candidates, two mechanism families and one
   complete economic experiment. Do not use already seen evaluation years to
   select a sign, formula, threshold or favorable interval.
6. Test source and mechanical identity, then a preregistered information and
   action gate, then a full native Qlib own-state account if admitted. A
   discrete local value screen is not a continuous SOXL/SOXX/cash allocator.
   Keep training and inference in the same risk domain; check whole shares,
   cash, Tiered fees, dividends, recovery, next-bar fill and terminal unknown.
   A failed fixed version gets one necessary attribution, then closes. Continue
   the broader Goal only with a materially different, falsifiable mechanism or
   a proven method defect; otherwise state the exact `needs_data`,
   `needs_distinct_decision_time_mechanism`, `not_promoted` or `blocked` reason.

The frozen f696 allocation plus b472 action/risk/recovery chain and its matching
Qlib own-account are attribution baselines for Stage1-derived research. Verify
the five f696 source outputs when substituting into a raw cashbook; the raw
cashbook's 27baa proposals are a different parent. The independent XGB/LGB
route is separate and did not pass its gates. Old beta/SOXL/cash limits belong
to their frozen routes and are not mandatory new-candidate constraints; simply
removing the beta cap already failed. Review the exact closed-route scope and
point-in-time clock before fitting. See `docs/stage1_v5_candidate_admission_20261030.md`
and the experiment cards for detailed negative evidence.

## Evidence, code and reporting

- Qlib is authoritative for alignment, Dataset, Workflow, Recorder, account
  analysis and conclusions. Use the repository `qlib_venv`; consult
  `.codex/skills/soxl-qlib-experiment/SKILL.md` and official Qlib docs for
  substantial Qlib changes. Keep source identity, raw/adjusted price basis,
  first availability, maturity, calendar and NAS receipt with each result.
  A saved object or local file is not a verified NAS restoration.
- Research runs live in `research/<experiment>/<experiment>.ipynb` with a
  frozen plan and slim card (`workflow.yaml`, `summary.json`, `conclusion.md`,
  selected-model metadata). Predictions, reports and positions belong in the
  Recorder. Reusable code lives under `src/`; `soxl_backtest/` is legacy
  compatibility. Do not create a second runtime hierarchy or put project state
  in a skill.
- State the exact SOXX verdict first, then matched-parent attribution, dates,
  costs, risk, good and bad intervals, missing evidence and next action. Keep
  raw factor information, fitted information, action value and account success
  distinct. Reused 2023–2026 history remains development evidence even when a
  new card or reviewer is used. Preserve original gates and conclusions of
  frozen historical cards.
- If a process is launched, its owner records a finite task, process identity,
  progress and terminal receipt. Observe that handle only while needed; report
  phase change, completion or concrete failure. A state file or old Goal status
  does not prove a live process or client Goal state. No perpetual model watch,
  heartbeat-triggered restart, or replay of the full research history into
  each task. Client Goal status requires a current client readback.

## Safe workspace changes

- At task start inspect `git status --short`, `git worktree list --porcelain`
  and `git branch --all`. The primary checkout and `main` are defaults. Preserve
  unrelated modified and untracked work; never bulk `clean`, `reset`, stash,
  force-remove or overwrite it. Stage exact paths. If isolation is necessary,
  integrate and synchronize the primary checkout before handoff.
- Before deleting research code or data, check imports, notebooks, registries,
  frozen manifests, serialized paths, production and NAS restore. Record exact
  bytes/hash, source commit and recovery path. Actually delete confirmed-useless
  detail; retain conclusions and evidence needed to reproduce or audit. If
  reference or restore safety is unknown, keep the object and state why.
- Use the smallest relevant verification once after the final change.
  Documentation-only edits need diff review and `git diff --check`; shared
  code changes get focused tests. Research success still requires native Qlib
  evidence and scientific gates. Preserve current/rollback release identities
  and run separate runtime checks before any deployment claim.
