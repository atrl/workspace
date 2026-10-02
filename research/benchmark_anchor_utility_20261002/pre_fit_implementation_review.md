# Independent implementation review — benchmark anchor utility

Scope: read-only source review before formal market fitting. No source edits, new forecast/model fits, portfolio replay or outcome-based changes were performed for this review. Result: **no blocking issue identified in the registered valid-data path**.

Reviewed in `/Users/rziiiii/dev/workspace-worktrees/benchmark-anchor-utility-20261002` against registered plan `4efdeb6e391c9b4a1d75c4ae995920e125dd862245b583f43f1075db2f3b9214`.

| File | Reviewed SHA256 |
|---|---|
| `src/model/benchmark_anchor_utility.py` | `327fb77dd441e8f4db27f138060e97aad3bb396b85c41989dfdd840b04317bf5` |
| `src/workflow/benchmark_anchor_utility.py` | `e6436ced18465aefdd3dd7f32e4ca65ac6de2e38867b9131b3afc27fd7dc5bc3` |
| `src/data/benchmark_anchor_utility.py` | `a6025810237efc7b958b3f66abf2bccf6a30f4144805d84ea117baed4d8cfa4e` |

## Priority checks

- **Current learner versus incumbent:** model lines388–448 differentiate `learner` against `current`, recomputed after each projected update. An improved incumbent is a separate deepcopy with its own saved evaluation. A worse current optimizer iterate does not borrow the incumbent's holdings/cache for its next gradient. Every candidate is scored by the full two-cost objective before incumbent acceptance; ties preserve the earlier/null incumbent.
- **Complete entry/fee scoring:** `_evaluate_alphas` uses all daily net-equity multipliers from CASH; no warmup exclusion is applied to J. The benchmark pays its entry in the same cost scenario. The scalar kernel debits risky purchase/sale fees from cash before asset returns; next-day holdings come from actual ending values. The fit's zero-relative-null self-check is not independent assurance by itself; reviewed tests additionally compare against independent `fee_affordable_target` + `settle_target_session` and analytic SOXX buy-and-hold entry wealth.
- **Global utility and adjoints:** fixed63session risk groups, global63/N mean, global RMS denominator/group count, and symmetric exact worst-cost ties match the plan. Final full risk-group overlap is retained and measured. Chunk code scores each daily row once, applies globally computed daily weights, and never chooses the worse cost or recomputes RMS locally per chunk.
- **Detached carried state:** each chunk begins from the current full policy's actual cost-specific pretrade holdings at its context boundary; only this boundary is detached. Up to21context sessions propagate local state gradients; the first scored chunk starts from CASH. Padded positions cannot alter holdings or objective counts. Gates are recomputed with autograd, not taken from detached cached alphas. Full scalar/Torch path parity is checked before backward.
- **Ensemble semantics:** hidden gates have no holdings input, so mean alphas produce exactly the mean target on the SAME current holdings. All three seed models participate; ensemble utility is audited on its own full cost-specific book. A negative ensemble J selects the exact null, never a successful subset or an average of seed returns.
- **Training provenance:** workflow lines74–98 bind each variant/year to scaler hash and prepared feature/return byte hashes; worker metadata must match both hashes before model/checkpoint persistence. The saved fold inputs retain native maturity/date hashes. Selected parameter tensor hashes are preserved by serialization; reviewed model tests check selected checkpoint identity and deterministic seed reproduction.
- **Evaluation causality:** replay binds bundle variant/model year/scaler year/hash/cutoff, carries actual holdings across annual model changes, and applies an invalid observed volatility fallback only on that date. Later valid dates continue from that forced transition's real ending holdings. Bad unused volatility cannot invalidate constant/price controls. Invalid training volatility anchors the affected fitted fold and is explicitly disqualifying.
- **Development/diagnostic separation:** the development ranking is sealed before post2022portfolio paths. The FINAL registered plan includes both predeclared `annual_refit` and `frozen2022_parameters` diagnostics. In the frozen mode, allocation/scaler/ensemble choice and2023volatility head stay fixed; later features update causally. The plan, rather than earlier proposal wording, is authoritative.
- **Economic evidence:** the existing same-at-least2of3seeds-positive-at-BOTH-costs rule is reused; all seed×cost rows are required. The ensemble separately receives the full stability/risk/bootstrap gates, invalid-source/numerical flags, and the1e−8numerical-null exclusion. Source/code/plan hashes are rechecked before final Recorder artifacts are accepted.

## Verification scope

I ran the15focused data tests successfully and independently checked the qualified native source against archived Phase4 Recorder `9a447c33dc934baeb1d4c3f2b9bf2122`. Using archived repetition1forecasts, every2016–2026annual preparation succeeded; native daily-maturity counts were1256,1508,1759,2010,2262,2515,2767,3018,3268,3520,3770. All33variant/fold scalers were eligible and produced finite Nx27 matrices, without any new forecast/model fitting. Details: `/tmp/benchmark_anchor_data_readback.json`.

The parent reports30model +15data +14workflow preflight tests, multiprocessing/Recorder roundtrips and synthetic profiling. This review inspected their relevant test logic but did not rerun those model/workflow suites or substitute code inspection for their execution evidence.

Known limitations remain deliberate and disclosed: nonsmooth fee autograd is not a certified one-sided derivative;21session state-gradient truncation is approximate; exact incumbent values do not certify a global optimum; the mixture class is restricted; constant masked seeds can have identical effective paths. No historical result, fresh forward evidence, actual execution quality or stable-excess claim follows from this implementation review.
