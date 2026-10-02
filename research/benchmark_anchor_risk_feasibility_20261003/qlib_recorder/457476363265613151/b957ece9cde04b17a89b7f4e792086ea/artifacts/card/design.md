# Cached DEVELOPMENT checkpoint feasibility diagnostic

Registered diagnostic design. The card selection_plan.json and initial executable notebook were created before the new intermediate-checkpoint feasibility evaluation; that plan is authoritative. No market fitting, changed checkpoint selection or alternative OOS policy path belongs to this task. The already known selected-incumbent tail failures motivate this audit; the history is not new unseen evidence.

Card: `research/benchmark_anchor_risk_feasibility_20261003`.
Parent: FINISHED Qlib Recorder `52f40778cbe64ca6a5864da356b960d0`, experiment `benchmark_anchor_utility_20261002`.
Parent plan SHA: `4efdeb6e391c9b4a1d75c4ae995920e125dd862245b583f43f1075db2f3b9214`.
Parent code commit: `f5cea33f6b3de0e6fc647bfadf4435f4e79343e3`; parent artifact-manifest SHA: `8aff34f5654d89ff406d0b2993500b5639529d8d31dec2d1e1702a1f973c27dd`.

## The bounded question

Do the existing saved optimization trajectories contain any checkpoint that simultaneously satisfies the **original** SOXX MaxDD/CVaR5 gates at10/25bp and has nonnegative original full-history centered training utility? This asks about the current optimizer's existing trajectory before changing the optimizer or risk objective.

A feasible checkpoint is not thereby an investable or qualified model. Do not choose a new policy, a successful seed, a subset of seeds, or a new ensemble. Do not evaluate development-year/OOS returns for the diagnostic epoch labels. The parent strategy selection and failure conclusions remain unchanged.

## Exact inventory and observation unit

- Three variants: constant, price, price_vol.
- Seven strict-prior fit cutoffs: January1 New York in2016…2022.
- Three fixed seeds:17,41,71.
-121saved learner epochs per run:0(interior initialization),1…120.
-63runs ×121 = **7623learner checkpoints**, yielding15246cost records.
- Add63explicit SOXX null controls, one per run, yielding126additional cost records. A run therefore has122checkpoint-status rows and244cost-metric rows, but learner feasibility denominators exclude its null.

Identical seed paths/checkpoints must not be collapsed. These are repeated, highly dependent training histories and parameter iterates, not7623independent tests. Missing, repeated, malformed or tampered epochs stop the diagnostic; do not silently reduce the inventory.

## Qlib source and identity chain

Read only the parent's `inputs/development_fold_inputs.pkl` and63known `fit_checkpoints/{variant}/{year}/{seed}/training_audit.pkl` artifacts, plus parent registration/source-hash metadata. Prefer these exact development artifacts over the aggregate training-diagnostics object that also contains later diagnostic-year models. Do not load portfolio-replay/OOS artifacts for this task.

Before deserialization, identify each artifact through the parent Recorder, record its canonical local bytes hash, and verify it against any parent artifact manifest plus a stable before/after read. Use `R.load_object`, not scattered CSVs. Retain the object/path/hash inventory in the new Recorder. Never fall back to the latest run, a CSV, mutable provider data, or fresh model fitting.

For each run validate:

1. Parent status/plan/code identities and exact variant/year/seed/epoch set.
2. Original fold date order, contiguous signal/trade/exit alignment, minimum252rows, feature/outcome availability strictly before its annual cutoff.
3. Every epoch's run metadata feature/return hashes equals the original saved fold matrices. Preserve their already standardized/masked27columns; do not refit a scaler or add H5 purge.
4. Scaler self-hash/schema/order and the source row/date/audit identity.
5. Each serialized checkpoint's tensor SHA, float64 architecture and globally feasible θ. Deserialize, never project/repair a saved checkpoint.
6. Stored training spec equals the parent's registered risk-group/cost/utility settings.

Use the `epochs`/`history` list only once. If both aliases exist, check their epoch/tensor/J identities agree; they are not two trajectories. Mark the original parent's selected epoch only as provenance.

## Computation, with no optimizer

For each saved learner checkpoint, deserialize the exact tensor snapshot and run the unchanged parent forward accounting over the **entire original training history from CASH** at both10and25bp. Include every return, initial entry and later fee; no warmup mask, position reset, terminal liquidation or shortened path.

Recompute the parent's full utility:

`x_t,c = log(g_policy,t,c) − log(g_null,t,c)`

`U_c = 63/N * sum(x_t,c) − sqrt(mean_b(max(−sum_group_b x_t,c,0)^2)+1e−12)`

`J = min(U10−U0_10, U25−U0_25)`.

The63session group starts and optional final full group must exactly match the parent. Context21belongs to the earlier gradient approximation and has **no masking role** in this forward audit.

Compare recomputed J and both centered U values to the stored epoch values using a fixed absolute1e−10utility tolerance. Preserve raw differences and any sign disagreement even within tolerance. Do not relax the tolerance after observing a mismatch. A material discrepancy is an integrity failure, not a risk-infeasible financial observation.

Independently build the same-date/cost SOXX `benchmark_path` and reconcile its daily wealth multipliers to the parent null kernel within1e−10. The null is separately labelled, not forged as a saved optimizer epoch. Cache reference arrays only when date hash, return hash and cost all match; record that deduplication.

Analyze policy and reference using existing `paired_metrics`, including native Qlib `risk_analysis(N=252, mode='product')`, initial-NAV anchored MaxDD and the existing exact fractional-tail CVaR5. Do not substitute unanchored/native-default drawdown or quantile-threshold CVaR with overweighted ties.

## Fixed feasibility definitions

Four original percentage-point comparisons:

- DD10 ≥ SOXX_DD10 −1e−10.
- CVaR10 ≥ SOXX_CVaR10 −1e−10.
- DD25 ≥ SOXX_DD25 −1e−10.
- CVaR25 ≥ SOXX_CVaR25 −1e−10.

Keep three separate checkpoint labels:

1. `risk_feasible`: all four risk gates, regardless of J.
2. `risk_and_J_nonnegative`: risk_feasible and raw recomputed J≥0.
3. `risk_and_J_strict_positive`: risk_feasible and raw recomputed J>0.

No negative-J tolerance rescue. A raw positive number near numerical precision is not itself economically material, an OOS return, or significance evidence. Do not silently replace J>0 with the parent's1e−8whole-path log-excess floor: the latter has a different metric definition. The registered plan additionally reports `risk_and_positive_J_margin = risk_feasible AND centered J>1e−8`. This is a separate numerical-margin diagnostic, retaining all three definitions above. It is expressly not a net-log-excess, profit or qualification gate.

For each63run, report complete counts, first epoch in each class, and maximum raw training J among risk-feasible and among nonnegative-J-feasible learner checkpoints. An exact tie uses the earlier epoch. Missing classes use null values, not epoch0. These are diagnostic labels only: output no executable “selected model,” ensemble or deployment bundle. Keep the guaranteed null outside learner counts.

## Minimal module interface

One new reusable workflow module is sufficient; do not change parent model/data/analysis files:

```text
load_development_checkpoint_source(source_spec, registration)
    -> frozen_folds, declared_runs, source_manifest

validate_checkpoint_inventory(frozen_folds, declared_runs, registration)
    -> checkpoint_index, integrity_audit

evaluate_checkpoint_fit(features[N,27], returns[N,2], dates[N,3],
                        training_audit, variant, model_year, seed, parent_spec)
    -> checkpoint_cost_metrics[244], checkpoint_status[122],
       run_summary, reproduction_audit

summarize_feasibility(all_checkpoint_metrics, all_checkpoint_status)
    -> run_summaries, variant_year_counts, integrity_summary

run_diagnostic(root, card, source_spec, workers)
    -> new Recorder identifiers and small final summary
```

The per-fit API evaluates exactly one saved121epoch trajectory plus its null. It cannot accept a requested winning epoch or OOS dataset. It must not call `fit_policy`, optimizer methods, variant search or future-year loaders. Pure synthetic fixtures can use shorter declared inventories for tests; the real run validates the full registered counts.

Initialize any required Qlib configuration **before** opening the new writable Recorder. Load immutable source objects through a read-only parent Recorder context, then use a separate new experiment/Recorder for outputs. Do not reinitialize Qlib inside an active Recorder or repeat the earlier lifecycle failure. Workers are read-only compute; one orchestrator writes artifacts.

## Storage and performance

Use63run-sized tasks rather than one process per checkpoint. Preload a run's immutable feature/return arrays and121small tensor snapshots, reuse its benchmark reference, and release daily policy arrays after extracting metrics. Never retain all7623full daily paths simultaneously. Normal paths are exactly reproducible from the parent input/snapshot/hash index; only an integrity-failing path needs separate full retention for debugging.

All15246learner cost rows,7623learner joint-status rows,126null cost rows,63null status rows, complete utility reproduction diagnostics and source manifests belong in the new Qlib Recorder. Root-side notebook, plan, workflow, summary/conclusion, compact counts and63run summaries remain small. No large predictions/positions/portfolio CSV in the card root.

No replay cache or cross-run resume is used. Persisted completed-run chunks are evidence only and are never reused as inputs to a later execution. After a failed execution, a fresh new Recorder evaluates the complete registered inventory again from the same hash-verified parent objects. Only in-memory benchmark reuse with identical date/return hash and cost is permitted. Final assembly requires all unique expected keys; no partial summary may be called complete. The parent frozen fold objects are referenced by their verified immutable Recorder hashes rather than duplicated into this card.

## Pre-fit verification and final interpretation

Before any market-checkpoint replay, synthetic tests cover exact null/entry fees, four risk-boundary comparisons, Jnegative/zero/positive labels, null exclusion, missing/tampered epoch failure, stored-U/J mismatch, final overlapping risk group identity, and deterministic diagnostic first/best labels. Check that no fitting/OOS entrypoint is reachable. Profile synthetic work only; do not change numeric definitions or choose checkpoints for speed.

After execution, assert the full inventory, stable source/code/plan hashes, all utility parity checks, independent benchmark parity, and Recorder readback. The notebook ends with a Chinese conclusion distinguishing:

- No saved learner checkpoint is risk-feasible: the cached trajectory did not reach the required region; the null alone remains feasible.
- Some risk-feasible checkpoints exist but their J≤0: feasibility is available, useful utility improvement was not found on this trajectory.
- Some jointly risk-feasible checkpoints have J>0: the trajectory contains training-feasible alternatives that the original unconstrained incumbent criterion did not retain. This still does not validate any of them out of sample.

Do not change the parent incumbent, choose a new strategy, infer independent evidence from epoch counts, or test alternative OOS returns. This pure training diagnostic does not add a return-hypothesis family, so the registered campaign count remains17. Any later feasibility-aware checkpoint rule or optimizer change is a **new separately preregistered policy experiment**, not a retroactive promotion from this audit.
