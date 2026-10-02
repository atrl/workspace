# Proposal: benchmark-anchored realized-sequence utility

Status: design only, before any new market fitting or portfolio replay. This document proposes a test; it does not claim profitable allocation, global optimizer convergence, or new independent evidence. The completed Phase 4 parameters and evidence stay unchanged.

## The specific question

Can a small, cost-aware allocation network learn a useful departure from an **exactly attainable SOXX benchmark**, when it evaluates the subsequent realized holdings/fees over an entire sequence instead of repeatedly valuing a five-day static asset holding?

This combines two explicit engineering hypotheses: repair the inaccessible softmax benchmark boundary, and evaluate actual repeated trading rather than a static H5 planning reward. It is not a clean estimate of either repair in isolation. The three information ablations below ask whether market/volatility information helps beyond a learned constant adjustment rule **within the repaired architecture**.

The new qualified-source moment audit already found that H5, 5μ1 and `(1+μ1)^5−1` all favor SOXX per estimated daily volatility at every 2016–2023 cutoff. Merely replacing H5 by μ1 is therefore not supported as an asset-preference reversal. Parent-owned decision evidence separately indicates that immediate transition fees can preserve a cheaper SOXL/cash position; its exact Recorder/artifact must be linked in the eventual registration. A longer static planning horizon could hide that fee issue without modeling future turnover. This proposal does not search that horizon.

## What is reused, and what is different

| Existing evidence | Reusable part | Failure/limit that this proposal must not conceal |
|---|---|---|
| `relative_wealth_lfnn_soxx_v1` | Long-sequence relative-wealth training concept | Softmax weights; 63-session **rebalance interval** and three-year bootstrap paths; target-relative-wealth loss; validation distribution failed. This is different from daily decisions with a 63-session scored objective. |
| `direct_continuous_soxx_allocation_v1` | Small direct network, annual walk-forward | Selected manual loss/cap profiles; 25bp and drawdown concerns. No claim that direct neural allocation is new. |
| `holdings_aware_pit_allocation_v1` | Actual holdings and exact order-turnover concerns | Fewer trades did not preserve the required risk/wealth results, especially after controller integration. |
| `stable_excess_integration_20261002`, Recorder `b94059c2e0754b0ab102e4f62534eefb` | Existing 63 scored sessions +21 warmup, exact differentiable cash-funded settlement, own holdings, small width8 network | This already tried realized sequence utility. High returns came with very large drawdowns/beta; all stable-excess gates failed. Old mixed price basis also limits interpretation. Long utility alone is not the new contribution. |
| `benchmark_risk_excess_20261002`, Recorder `bba55e94a1d04b4a9c68dca562766159` | Feasible-benchmark training diagnostic | All21 constant annual/seed models had loss0.00459–0.01735 versus exact SOXX≈0.000001. Finite softmax parameters could not represent the boundary; weight decay was not shrinkage toward SOXX. |
| Phases3–4 | Exact funding/settlement, benchmark null audit, qualified price basis, full path gates | Static H5 reward pays only the immediate planning transition while the actual policy reoptimizes daily. Extending H5 until fees look small is not evidence of correct long-run behavior. |

These distinctions are based on the named inspected branches, not a claim that every historical attempt has been exhaustively ruled out. The old cards cannot be ranked against the new basis/windows by their raw returns.

Primary code references: `src/model/integrated_allocation.py:61` settlement kernel; `:199` chronological rollout; `:287` block construction; `:299` portfolio objective; `:369` AdamW; `:395` training metadata. Exact independent settlement: `src/backtest/position_accounting.py:8`. Old network: `src/workflow/direct_continuous_soxx_allocation.py:126`. Prior target-relative-wealth objective: `src/objective/relative_wealth.py:88`.

## Recommended minimal design

Use **one fixed interior learner initialization for each of the same three registered seeds**, and keep an independently audited θ=0 benchmark incumbent. Use market-only hidden gates, actual holdings only through the HOLD expert, and no explicit cost input in this first test. Both10/25bp costs still enter training and exact forward accounting. This is sufficient to test the proposed repair without a hidden optimizer grid, cost-conditioned policy, or zero-head initialization deadlock.

Source remains the qualified native provider `data/qlib/published_futu_affine_20260925_v1`, as_of2026-09-25, parent Recorder `bd754e6362b241f194351a1e7aebff6f`. Use the Phase4 causal25features with `log(volume)` z-scores. Features, forecast inputs, labels, annual cuts and exact artifact identities remain Qlib/DatasetH/Recorder backed. No changes to the completed Phase4 evidence.

## Feasible architecture and its limits

Asset order is `[SOXL,SOXX,CASH]`; h is actual pretrade net holdings. Fixed experts are `e_L=(1,0,0)`, `e_C=(0,0,1)`, `e_H=h`, and benchmark `e_S=(0,1,0)`.

`b_t=[1,sigmoid(W standardized_market_t+a)]`, width8, so b has9 bounded components.

`θ` is a3×9 nonnegative matrix with **one global constraint** `sum(θ)≤1`.

`α_j,t=Σ_k θ_jk b_k,t` and `target_t=(1−Σ_j α_j,t)e_S + Σ_j α_j,t e_j,t`.

Because `0≤b≤1`, `Σα≤Σθ≤1`; targets are long-only simplex allocations without borrowing. They are desired **post-fee net weights**. Reuse exact cash-funded settlement to obtain executable orders. Do not append normalization, a SOXL cap, cash floor, EMA, stop timer, regime threshold, or ex-ante variance projection.

At θ=0 the target and entire forward path are exactly SOXX at finite parameters when compared from the same CASH origin. An annual switch from a prior non-SOXX book still pays its actual liquidation/entry fees; it cannot replace that historical book retroactively with buy-and-hold. `θ_L,0=1`, `θ_C,0=1`, and `θ_H,0=1` express pure SOXL, cash, and HOLD. These are unit-test endpoints, not fitted candidate families. An exactly represented benchmark is not automatically an optimization guarantee; the incumbent audit below supplies the limited training guarantee.

The global θ budget deliberately restricts dynamic expressivity: near-full L in one state and near-full CASH in another generally require both coefficient budgets near1 and therefore violate the shared sum≤1. Pure constant endpoints remain possible; arbitrary state-dependent vertex switching does not. This is a small benchmark-departure class, not a universal allocator. Shared gates further restrict independent expert patterns. A failure cannot reject all allocation models.

With no holdings inside hidden gates, gates for every training date can be precomputed once per parameter state and reused at both costs. The HOLD expert still makes the action stateful; actual own holdings are updated after every trade and return. A constant-coefficient mixture with HOLD is a learned adjustment/inertia rule, not static weights and not a predetermined moving average.

## Three information variants

1. **Constant:** zero all hidden inputs. Actual HOLD remains h.
2. **Price:** the fixed25causal market features.
3. **Price+vol:** those25fields, one causal OOFvol5 prediction, and its availability indicator. No cash-return prediction input.

Instantiate the maximum27input width,8gates and27θcoefficients for all three; mask disabled channels to exactly zero after train-only scaling. This gives the same nominal architecture (216W +8bias +27θ=251parameters), but not identical effective capacity: masked directions cannot learn. Use that honest description in the registration.

Price scalers use prior training data only. Standardize the volatility prediction on available prior training predictions, neutralize unavailable predictions after scaling, and retain its explicit0/1availability flag. Preserve2011+history instead of dropping pre-OOF rows. Nonfinite/nonpositive output from an available volatility model is an invalid source, not normal warmup; retain the existing explicit failure/disqualification policy. Use the already fixed Ridge10/common16prior-only forecast protocol, with no head/window search.

No explicit cost feature is recommended initially. Otherwise the10/25bp comparison becomes partly a test of knowing which cost scenario applies, adding another interpretation and allowing different gates. Cost-blind gates plus independent fee-funded holdings still allow the learned strategy to respond to the consequences of costs through HOLD. A cost-conditioned extension would require its own clear preregistration.

## Optimizer: one interior start, audited null retained

For each of the fixed three seeds, initialize hidden W with the fixed PyTorch Linear-style uniform distribution `U[−1/sqrt(27),+1/sqrt(27)]` and bias0. Initialize **every θ element to1/54**, giving total θmass1/2. This is the midpoint between the origin and the uniform simplex boundary center: all coordinates receive gradients and half the coefficient budget remains slack. It is an algebraic initialization, unrelated to return outcomes. This is one data-independent interior optimization start, not an allocation floor, a production risk preference, or an epsilon selected by performance. θ may subsequently reach zero or any feasible boundary. Do not search multiple masses or select successful seeds.

The θ=0 policy is separately evaluated and saved before training. It remains available even if the interior initial model or every subsequent iterate is worse. Initialization is thus explicit rather than a softmax's unavoidable uniform allocation. It is still an optimization choice whose influence must be disclosed.

Reuse the fixed120updates, learning_rate0.003 and gradient clip5 unless the final registration explicitly fixes another engineering schedule before fits. Use **projected Adam with no parameter weight decay**: the previous AdamW decay was not benchmark shrinkage. After each proposal, project flattened θ onto `{θ≥0,sumθ≤1}`. If `sum(max(v,0))≤1`, keep that positive part; otherwise use the unique τ satisfying `sum(max(v−τ,0))=1`. Validate finite values and feasibility after every projection. Do not claim global convergence or an exact constrained optimum for projected Adam.

At θ=0, `∂target/∂θ_jk=b_k(e_j−e_S)`, but `∂target/∂W=0`. The utility derivative may also be unhelpful or zero, and HOLD equals SOXX after benchmark entry. Consequently a zero-initialized joint learner can remain stuck with no learned gates. The fixed interior start avoids *that particular initialization deadlock*; it does not guarantee information, improvement, or escape from later local optima. Record gradient norms, θmass/active coordinates, anchor outcomes and selected epochs for every seed/fold.

## Exact forward accounting is essential; boundary autograd is not certified

The reused settlement kernel is forward-correct but nonsmooth at no-trade. It uses `abs(trade)` and detached selection among four sign roots. Equal forward roots at a boundary need not give the same derivative.

A synthetic read-only probe (no market data/fitting) found: with pretrade100%SOXX, zero asset returns, target `θL+(1−θ)S`, existing autograd at θ=0 reports `d log wealth/dθ=0`. The correct feasible right derivative is `−2c/(1−c)`: −0.002002002 at10bp and −0.005012531 at25bp. For CASH departure it is `−c/(1−c)`, while autograd again gives0. A1e−7one-sided finite difference matched the analytic results. This is a boundary derivative limitation, not an allegation of incorrect forward P&L.

Do not introduce a fake straight-through derivative, smooth away actual fees, or call every returned autograd value a verified one-sided gradient. Nonsmooth subgradients may generate optimizer proposals; exact full-objective incumbent acceptance determines what is retained. This protects the reported training incumbent, not optimizer convergence. Synthetic one-sided and away-from-kink derivative checks remain required, with the boundary limitation recorded rather than hidden.

## Full chronological objective: every return and every fee

For every parameter state used in an incumbent comparison, run the **entire eligible training history once from CASH** at each cost. The SOXX reference also starts in CASH, pays the same fee-funded entry, and then holds. All daily returns, entry fees and later fees are included. No21session cost mask, yearly restart, free established position, or terminal liquidation is introduced.

At cost c define `x_t,c=log(g_policy,t,c)−log(g_SOXX,t,c)`, with g from exact actual holdings and settlement.

Keep a fixed63session risk horizon. Define full63session groups at starts0,63,126,…, and, if needed, one final full group ending at the training cutoff. This preserves the inherited final-window coverage convention; disclose any overlap in that final risk group. All observations are counted **exactly once** in the global mean, even when the risk grouping overlaps.

`Q_b,c=Σ_{t in group_b} x_t,c`

`U_c=63/N * Σ_{t=1..N} x_t,c − sqrt(mean_b(max(−Q_b,c,0)^2)+1e−12)`.

This retains the inherited negative-quarter RMS coefficient1 and63session scale, while replacing independently reset block wealth with a full continuous path. It is a declared objective-boundary repair, not an unchanged copy of the old training objective. There is no H grid. The final full group avoids extrapolating a tiny terminal stub into a synthetic quarter; any duplicated risk-window membership is auditable.

The benchmark has x=0, hence U≈−1e−6 from the numerical RMS convention. Center each cost's utility by its independently replayed anchor value before comparison. Preserve the epsilon convention as a fixed numerical choice, not a negative economic return or a tunable risk preference.

Train against the worst aggregate cost utility:

`J=min(U10−U0_10, U25−U0_25)`, where U0_c is the exact SOXX utility at that same cost.

Every cost has its own holdings/NAV path. Do not subtract25bp from a10bp replay, choose the worst cost on each day, or splice scenario states. Compute both scenarios even if25bp is usually worse, since holdings-dependent behavior need not be monotone in cost. The loss is the negative of this complete-history utility; a tie uses a declared symmetric subgradient. No smoothed-min temperature is needed.

The minimum is preferred to a sum/mean: both costs are required stress gates, and no probability such as50/50has been supplied for averaging acceptable25bp deterioration against10bp gains. The two centered utilities have identical unitless63session scaling, identical dates, risk groups, starting cash and matching-cost benchmarks, so this minimum is comparable. It is robustness to these two declared scenarios only, not to every cost/spread/impact process.

This differs from the older10bp-only fit. All three new information variants receive identical two-cost training, so their internal comparison is fair; their comparison to older models does not isolate architecture alone.

## Fast gradients without free boundary states

Use a scalar Python/NumPy full chronological forward kernel with precomputed gates for exact carried states and incumbent values, plus batched Torch chunks of at most84steps for gradients. No Numba dependency is assumed. Profile this on synthetic data before market fits; verify full-path parity against the independent exact settlement. The following truncation is a fixed computational approximation, not another policy family:

1. At the current parameters, precompute all market gates and run both full chronological cost paths, retaining true pretrade holdings at every date and the exact complete J.
2. Compute the global utility derivative with respect to each daily log-equity multiplier, including the active worst-cost branch and all63session risk-group memberships.
3. Cover the training rows with disjoint scored chunks of63 (the last gradient chunk may be shorter). Start each from the **same current-policy carried holdings** at its context boundary, detached for differentiation; allow up to21prior sessions of differentiable context, inherited from the old warmup length. The first chunk starts from real CASH and scores its entry.
4. Use the globally derived per-date objective weights, not a separate mean/min/RMS recomputed inside each chunk. Immediate losses are scored once;21session context exists only to transmit state gradients. It never masks fees from the actual objective.
5. Accumulate the full gradient before one optimizer update, so carried states do not become stale between partial updates. Recompute exact full paths after the update for the incumbent audit.

For unambiguous risk normalization, let `D_c=sqrt(mean_b(min(Q_b,c,0)^2)+1e-12)` and M be the **global** number of63session risk groups. Then

`dU_c/dx_t,c = 63/N − (1/(M D_c)) * Σ_{b containing t, Q_b,c<0} Q_b,c`.

Apply weight1to the strictly worse cost branch,0to the other, and1/2to each exact tie. The minimization loss uses the negative of these weights. Neither N, M, D nor the active cost is recomputed per gradient chunk. The weighted log-equity sum is only a first-order gradient surrogate; incumbent selection always recomputes the original nonlinear global J. Explicit chunk padding carries a false validity mask and cannot trade, earn a return, or enter any objective count.

Truncated state gradients are an approximation: future holdings effects beyond the context may be missed. Full forward values and incumbent utility must nevertheless match an independent continuous replay exactly. Benchmark and all three variants use the same gradient schedule. Profile this engineering path before launching fits; do not choose truncation length, number of starts or training schedule using P&L.

## Incumbent, checkpoints, and ensembles

Save the complete θ=0 model as incumbent with centered J=0. After each registered update, evaluate the full training path at both costs and retain the complete candidate parameter state only if the exact centered J improves. Ties retain the simpler/earlier incumbent. The optimizer trajectory itself may visit worse models; “never below the benchmark” refers only to the saved **training-utility incumbent**, not every update, every year, drawdown, or unseen returns.

Because centered `J=min(J10,J25)`, J≥0 already implies no worse training utility than SOXX at either cost; audit and report both components explicitly. Do not select checkpoints using sampled gradient chunks, development P&L or diagnostic years.

Keep all three fixed seeds; no winning-seed selection. If the evaluated policy is an equal-target ensemble, every constituent must see the same ensemble-owned holdings, and the ensemble needs its **own exact full-training comparison to θ=0**. Individual-seed utility guarantees do not automatically transfer through averaging and state-dependent HOLD. If the ensemble fails that training criterion, its fitted result is the null; do not choose a successful subset of seeds.

## Annual cuts, frozen test, and selected-ensemble semantics

- Fixed seeds are17,41,71, matching the original stochastic-seed protocol. For each variant and every2016–2022fit year, initialize the learner **fresh** at the same interior θ and the corresponding fixed hidden seed; reset optimizer moments and fit prior-only scalers. Do not warm-start parameters, gradients or optimizer state from the previous fold.
- Fit cutoff is January1 00:00America/New_York. Eligible primary signals start2011-01-03; `feature_available_at` and each scored one-day return's actual `return_available_at` must be strictly before cutoff. Require at least252eligible daily rows and sufficient full63session risk groups. No later/unknown labels, forecast targets or immature rows may enter the objective; no missing row can be silently deleted to create an apparently contiguous path. The unused H5 target must not unnecessarily shorten an otherwise mature daily-utility training sample.
- Volatility-head targets use their own stricter `forecast_available_at` maturity. Training policy inputs use prior-only annual OOFpredictions, and eligible pre-OOFhistory remains with neutral value/flag0. Forecast availability is not inferred from a later model that happens to cover the date.
- Each annual fold's seed incumbent is selected only by exact complete training J. Build the fixed equal-target three-seed ensemble, evaluate its own full training path from CASH at10/25bp, and compare it with exact SOXX using the same J. If it is worse, the **fold's selected ensemble policy** is the null; retain every seed/checkpoint and the rejected ensemble audit. Do not use a best seed or subset in its place.
- Development is one continuous2016–2022signal-year book per cost. Annual model changes carry that book's actual holdings and charge actual transition fees; do not replace them with training terminal holdings or reset them to CASH each January. The benchmark and each individual seed have their own continuous cost-specific books.
- The reused2023+diagnostic uses a policy/scaler trained strictly before2023-01-01, and its fixed2023volatility head; no2024+refits or test-driven warm-starts. Start the diagnostic candidate and benchmark from the same CASH origin as the existing diagnostic convention, then carry continuously through all observed mature daily labels to as_of2026-09-25. This single separate origin is not an annual reset. Do not choose annual-refit versus frozen policy after viewing the diagnostic.
- Keep exact signal/trade/exit date pairing.2016–2022qualification includes the already specified final settlement into early2023 for the last2022signals; the next diagnostic holding interval must not overlap that last interval. Terminal as_of inference rows with unavailable next-open outcomes remain out of performance metrics, not assigned zero returns.
- Variant ranking and any diagnostic selection follow the preregistered development gates, independent of checkpoint/seed selection. If no variant qualifies, report the diagnostic ranking and all failures without claiming promotion. The audited null is still a legitimate training outcome, but zero excess does not pass positive-excess qualification gates.

**Preserve the original stochastic-seed gate:** the same at least2of the3declared individual seeds must have strictly positive cumulative net log excess at **both10and25bp** on the full development path. Two different successful seed sets at the two costs do not satisfy this. Save all6seed×cost outcomes; do not drop a failed or missing seed. This is the original `StabilityGateSpec.minimum_positive_seeds=2` contract in `src/analysis/integrated_allocation.py:233–249`, not Phase4's3/3deterministic recomputation identity check. Report every seed's tails/yearly behavior as well; the ensemble must separately pass the complete unchanged economic gate set.

## Nonfinite results and preregistered falsification

Invalid source features, impossible returns≤−1, nonpositive NAV, nonfinite parameters/preactivations/gradients/objective, failed feasibility, or disagreement between scalar and Torch forward accounting are numerical/data failures. Stop the affected fit and retain the failure; a baseline accounting path may be recorded for completeness but cannot convert a failed seed into a valid fitted candidate. Do not silently replace data, shrink θ, add a cash floor or omit a seed to pass the gate. A deliberately selected finite null and a nonfinite-failure fallback are different outcomes.

Before the first market fit, fix these interpretation rules:

1. If Θ0 cannot reproduce the benchmark or chunk/full-forward values do not reconcile, reject the implementation and run no market fit; this is not evidence about predictive information.
2. If all finite learners end with the null incumbent, report no improving departure found in this class/optimizer, along with hidden/outer gradient and movement diagnostics. Do not claim universal absence of alpha and do not add unregistered starts, horizons or widths.
3. If price or price+vol fails to improve on the constant-coefficient control under both-cost economic gates, there is no demonstrated benefit from the added conditional information in this test. A higher raw terminal wealth accompanied by more beta or worse tails does not overturn that result.
4. If training utility improves but development stability/risk/seed gates fail, reject promotion as a generalization/economic failure. Do not use the reused2023+diagnostic to choose checkpoints, repair thresholds or select seeds.

## What this fixes, what it cannot fix

This removes the finite-softmax inability to reach the benchmark, makes the protected null explicit, avoids the zero-head initialization deadlock, and gives transaction costs their actual repeated-path consequences instead of dividing an immediate fee by an arbitrarily enlarged static planning horizon.

It does not create forecast information, certify optimization, erase source uncertainty, or turn reused history into new evidence. The global mixture class is restricted; hidden gates have no direct holdings input; truncated gradients can miss long state effects. These are disclosed tradeoffs for the first bounded test.

Removing the ex-ante covariance cap restores the possibility of very high SOXL risk. Phase1 already showed that high terminal wealth can be purchased through more beta and extreme drawdown. Keep the exact same realized SOXX benchmark,10/25bp, annual-stability, MaxDD, CVaR, worst-day, turnover, paired uncertainty and beta/HAC/risk-matched diagnostics. The63session downside utility is a learned training objective, not a substitute for those gates. Cash still earns0; do not add an extra financing charge to ETF returns or silently add a cash yield advantage.

## Required pre-fit contracts

- Θ0 exactly matches independent fee-funded SOXX entry and continuous NAV at both costs, including serialization and ensemble inference.
- Projection, simplex, all expert endpoints, invalid-input failure, real HOLD/no-order, and actual cash funding are tested.
- The synthetic boundary derivative discrepancy is documented; away-from-kink and short-sequence gradient checks distinguish exact forward values from approximate/subgradient optimization.
- Full objective and gradient-chunk reconstruction reconcile on a short deterministic fixture; entry and any warmup/context fees cannot disappear. A chunk boundary must carry true current-policy holdings, not CASH or a target fabricated from future data.
- Every scored daily outcome is known before its annual fit cutoff. Price and forecast scalers are prior-only; vol inputs use the fixed OOF protocol. No calendar resets in development or the frozen2023+diagnostic path.
- Incumbent audits include all training rows and both costs; seeds/epochs/null outcomes and numerical failures are retained. Anchor-only outcomes are not claimed as positive alpha evidence.
- No notebook/Recorder/cache may silently reuse old mixed-basis inputs, oldsoftmax outputs, or an old10bp-only objective.

## Research accounting and decision

Three information variants add three registered hypotheses: if the current campaign family is14, make the explicit denominator17. That adjustment still does not cover every historical/adaptive attempt. Pure SOXX is a mandatory null, not a selected alpha candidate. New-basis2023+history remains already observed.

A useful positive result must first pass the architecture/accounting contracts, then show that conditional information improves on both the exact SOXX benchmark and the constant-coefficient HOLD control under unchanged full economic gates. If no fitted departure improves the audited training objective, retain SOXX and report the optimizer/capacity outcome honestly. If the model beats the constant control but fails tails or annual stability, it remains diagnostic.

This is deterministic differentiation through a fixed Qlib-managed return panel and a declared supervised allocation utility. There is no exploration environment, Bellman target, critic, policy-gradient estimator or online reward feedback. Sequential holdings do not recreate the retired RL reward flow. No outcome here promises future stable excess or authorizes real broker execution.
