# Risk-aligned whole-path training — proposal before new fitting

Status: proposal only. No new model fit, changed checkpoint selection, alternative OOS path or implementation has been run. Parent checkpoint-feasibility evidence is FINISHED Recorder `b957ece9cde04b17a89b7f4e792086ea`; original utility study is Recorder `52f40778cbe64ca6a5864da356b960d0`.

## What the cached trajectories establish

All7623development learner checkpoints and63separate exact-SOXX nulls were audited; source and original utility reproduction passed. Each family has2541learner checkpoints. The outcomes below are training diagnostics, not independent samples or a newly selected policy.

| Family | Four-tail-gate feasible | Feasible J≥0 | Feasible J>0 | Feasible J>1e−8 |
|---|---:|---:|---:|---:|
| constant | 594 | 381 | 339 | 336 |
| price | 36 | 0 | 0 | 0 |
| price_vol | 30 | 0 | 0 | 0 |

For price, risk-feasible counts by2016…2022cutoff are20,5,1,5,2,3,0; price_vol counts are17,4,1,5,2,1,0. Risk feasibility therefore existed at some early/intermediate states, but neither conditional family's cached trajectory contains an improvement over the null under the original J. This is not proof that the model class is infeasible or lacks all information.

The best feasible J is price−0.000205302 (2016/seed71/epoch22) and price_vol−0.0000143836 (2016/seed71/epoch23): about−2.053 and−0.144basis points of the **registered63-session utility**, not return basis points. Constant's feasible positiveJvalues range from2.32e−10 to4.01e−5;42feasible constant states have exactly zero θ. The three constant seed trajectories are effectively identical; their counts should not be read as independent evidence.

A crucial qualification: **positive cumulative training net excess does exist at some risk-feasible conditional checkpoints even though J is negative**:

- price:3states, all at2016cutoff; seed41 has1, seed71 has2.
- price_vol:4states, all at2016cutoff; seed17 has1, seed41 has1, seed71 has2.
- None of the2017–2022cutoffs has such a conditional state. Cached coverage is therefore only1/7training-cutoff years, not enough even to supply non-null candidates across5/7fit years. This is not an OOSyear-return claim, and no alternative ensemble was formed.

The exact conclusion is “no jointly tail-feasible improving conditional state under the old declared utility on these trajectories,” not “no feasible profitable state.” The quarterly negative-relative-return RMS veto, with coefficient1, excludes those few training-profitable/tail-feasible states. It encodes an additional preference that is not equivalent to the final annual stability or absolute tail gates.

## Two defensible objective choices

**A — preserve oldJ, add constraints.** Maximize the same worst-cost mean-minus-quarter-shortfall utility while requiring the four benchmark tail constraints. This isolates adding explicit risk feasibility and retains the earlier quarterly relative-shortfall preference. However, it deliberately continues rejecting profitable/tail-feasible states when their quarter-shortfall penalty dominates. That is defensible only if this extra utility preference is itself intended; the current user goal does not clearly authorize it as an additional profit veto.

**B — maximize worst-cost mean net log excess, add the same constraints; retain oldJ only as a diagnostic. Recommended.** This directly matches cumulative net relative wealth and the already established MaxDD/CVaR requirements. It removes the arbitrary quarterly RMS coefficient from the training acceptance rule. Annual stability, uncertainty, tail, seed and forward gates remain unchanged evaluation requirements; no training claim can substitute for them.

B is a joint objective-and-constraint alignment hypothesis, not a clean causal estimate of either change alone. It does not promise profitable feasible models or solve OOSgeneralization. Removing the RMS veto alone from old checkpoint selection is insufficient: cached conditional candidates with positive mean and feasible tails cover only the2016fit. The new test must train new trajectories under the aligned objective; it must not retrospectively deploy cached states.

## Keep everything else fixed

Use the same benchmark-anchored architecture (27market inputs,8sigmoid gates,3×9nonnegative θ with global sum≤1), actual HOLD expert, no holdings/cost input in hidden gates, same three information variants and seeds17/41/71. Keep fixed interior θ=1/54, standard seeded hidden initialization/bias0, fresh annual initialization, projected Adam without decay,120updates, primal learning_rate0.003, gradient clip5, scored gradient chunks63with up to21prior context, CPUfloat64.

Reuse the exact qualified Phase5source/feature definitions, original strict-prior training dates, scalers and OOFforecast contract. Prefer hash-verified original prepared inputs for the development comparison; do not introduce source changes or new forecast fits as another intervention. Keep all invalid-input/fallback disqualification rules. No new cash floor, SOXL cap, ex-ante variance budget, stop/timer, feature, sparsemax, extra hidden-state input or horizon grid.

## Recommended objective and scaling convention

For each cost c∈{10,25}bp, use its own full chronological training path from CASH, all returns and every entry/later fee. The reference is same-date, same-cost fee-funded SOXX buy-and-hold.

Under this exact accounting contract—cost-blind gates, desired POST-FEE normalized target z, continuous shares and proportional fees—the normalized ending holdings satisfy

`h_next = z*(1+r) / sum(z*(1+r))`.

The common funding factor cancels, so normalized holdings/target paths starting from the same state are normally cost-independent in exact arithmetic, apart from rounding. HOLD does **not** automatically adapt to cost through normalized weights in this model. Fees still change NAV, actual funded trade amounts, learning utility and net-return risk metrics. Both cost books must therefore be settled and compared to their matching benchmarks; equal normalized weights across costs are legitimate, not evidence of a copied fee path. This is a method-description clarification, not a change to prior accounting code/results. Each of the four dual variables below uses its own0.003step, even when some cost constraints are numerically similar.

`x_t,c = log(g_policy,t,c) − log(g_SOXX,t,c)`.

Define `A_c = 63 * mean_t(x_t,c)` and `F = min(A_10,A_25)`.

The63factor preserves the unit scale of the earlier objective's mean term. It is a fixed positive unit conversion, **not a planning holding period or searched horizon**; maximizing it has the same population argmax as mean daily log excess. Record both daily mean and aggregate log excess, so this scaling cannot masquerade as realized63-day returns. It affects finite-step optimizer/dual conditioning and is therefore explicitly preregistered.

The minimum is preferable to a mean across costs: both costs must succeed and no50/50scenario probability has been supplied to justify letting one compensate for the other. Both values have identical date membership, normalization, starting capital and matching-cost references. Exact ties split the objective gradient equally. No softmin temperature. Do not reuse the oldJ's worst-cost selector for this new F.

Positive F means positive cumulative training log excess at both costs. OldJ (including the original63risk-group RMS) is still computed and saved, but it never enters the primal objective, dual updates, incumbent acceptance or ensemble veto.

## Four unchanged empirical tail constraints

Let `D_c = −MaxDD_c/100` be positive compounded drawdown depth, including the initial NAV1mark. Let `C_c = −CVaR5_c/100` be signed lower-tail expected loss. A policy with all-positive tail returns may have C<0; do not clamp it. Use exactly the existing fractional5%tail-mass calculation.

The mathematical constraints are `D_policy,c ≤ D_SOXX,c` and `C_policy,c ≤ C_SOXX,c` at BOTHcosts. Audit feasibility using the original comparisons in percentage points with1e−10tolerance. Do not loosen those gates during training or introduce an epsilon risk allowance.

For the dual-update convention, normalize by the **fixed prior-training reference risk**:

`g_D,c = (D_policy,c − D_SOXX,c) / D_SOXX,c`

`g_C,c = (C_policy,c − C_SOXX,c) / C_SOXX,c`.

These positive denominators leave the feasible set unchanged but condition gradients and dual-step size. Require finite positive reference magnitudes for this chosen convention; if they are unavailable/nonpositive, fail closed instead of inventing a tiny floor. The registered source has nonzero benchmark drawdowns/lower-tail losses; the requirement should still be checked.

Use ideal g≤0 in dual updates. The original1e−10percentage-point allowance is only the established numerical gate tolerance (1e−12in return fractions); it is not an economic risk budget. Save raw percentage-point gaps AND normalized g.

## Deterministic primal/dual sequence

Use four nonnegative dual variables, one for each metric/cost. Start every fresh fold/seed with λ₀=0 and the same interior primal model as Phase5. Save exact-SOXX as a separate feasible incumbent with F=0.

Minimize the Lagrangian loss `L(q,λ)=−F(q)+Σ_i λ_i g_i(q)` over the unchanged parameter class. Freeze projected dual stepηdual=0.003 **for the normalized g convention above**. This numerical value is an optimization choice, not a financial constraint or a guarantee that120steps will converge; do not tune it after results.

For update k:

1. Compute both complete current-policy paths and the true F, raw risks, normalized g and global daily adjoints at q_k.
2. Hold λ_k fixed, accumulate one complete approximate Lagrangian gradient using the existing true-carried-state21context TBPTT and globally computed weights. Apply one projected-Adam primal update, clipping the combined gradient at5 and projecting θ onto its original feasible ball.
3. Evaluate q_{k+1} on both full paths. Update `λ_{k+1}=max(0, λ_k +0.003*g(q_{k+1}))`, using **signed post-primal** constraint values. Slack constraints reduce multipliers. No arbitrary λcap or positive-part-only accumulation.
4. Independently audit the new model for all four raw tail gates and F. Keep a deep-copied incumbent only if all gates pass and F strictly improves on its current value; ties keep the earlier/null incumbent. Do not rank states by changing Lagrangian values.

Archive λ_k used for the primal step, post-step g, λ_{k+1}, both Avalues, F, oldJ, risk metrics, feasibility, gradient norms, primal-step norm and selected-incumbent identity. Distinguish current learner/evaluation from the retained incumbent, as in the repaired Phase5code. No annual parameter or dual warm-start.

## Global risk adjoints and both cost branches

Let `l_t=log(g_policy,t)` and `L_t=Σ_{s≤t}l_s`, with L₀=0.

For drawdown, use the deterministic active peak i and trough j attaining `D=1−exp(L_j−L_i)`, with an initial-capital candidate and deterministic earliest tie convention. Away from a tie,

`∂D/∂l_t = −exp(L_j−L_i) * 1{i<t≤j}`.

If D=0, select the initial zero-depth pair and a zero subgradient. At tied peaks/troughs the selected subgradient is a declared numerical convention, not a unique directional derivative.

For exact lower-tail loss, set m=.05N. Give weight1/m to returns strictly below the boundary, and distribute the remaining fractional tail mass equally across all returns exactly equal to that boundary; all others have weight0. Weights sum to1and reproduce the existing CVaR value without chronological bias at a tied boundary. Then

`∂C/∂l_t = −a_t * exp(l_t)`.

Divide these derivatives by their fixed benchmark denominators for ∂g. No quantile smoothing, threshold clipping or surrogate tail fraction. Validate values against the existing anchored-DD/CVaR helpers and derivatives by finite differences away from ties.

For the mean-only objective, the utility adjoint is simply63/N on the active worst-cost branch (half on an exact tie). **Risk gradients from both cost paths must remain present even when that cost is inactive for F.** The loss adjoint is

`a_loss,c,t = −p_c*(63/N) + λ_D,c*∂g_D,c/∂l_t + λ_C,c*∂g_C,c/∂l_t`.

An explicit combined-adjoint wrapper may reuse the existing chunk kernel and its current-path parity checks; never mutate the original evaluation/utility record to make oldJappear to be the new objective. Compute normalization/active extrema/tail mass globally before chunking. Context rows transmit state derivatives only; every actual fee remains in the complete forward objective and constraints.

This retains two important limitations: fee gradients at exact no-trade are not certified feasible one-sided derivatives, and21session truncated state gradients omit longer holdings effects. MaxDD can span much longer intervals, so exact global loss values do not imply exact full-state risk gradients. No global optimum, KKTcertificate or guaranteed feasibility from primal/dual iteration is claimed. Exact full-path incumbent checks supply the narrow saved-training feasibility guarantee.

## Incumbent, ensemble, and failures

Always keep exactSOXX available. Save the best **F** among states passing the same four training tail gates; oldJnegative is not a veto. All three seeds are retained. Their equal-target ensemble needs its own complete two-cost mean/tail audit on ensemble-owned holdings; if it fails feasibility or F≥0, select the exact null for that fold rather than a subset of seeds.

Nonfinite source inputs, parameters, preactivations, gradients, λ, risks or utility; nonpositive NAV; failed projection/funding; or scalar/Torch parity disagreement are explicit failures. Keep null accounting only and disqualify the affected fit; do not cap λ, change step sizes, extend updates, clip features or omit seeds to rescue it. A valid null outcome is distinct from a failed-fit fallback.

Save all121primal states, all dual/constraint trajectories and optimizer state/provenance in the Qlib Recorder. Full daily arrays need not be duplicated for every checkpoint when immutable inputs/tensors permit replay; retain exact chosen-incumbent and ensemble paths and any failure evidence. Record peak/trough identities, tail-weight metadata or hashes, and raw/normalized constraints to make gradients auditable.

## Pre-fit verification and unchanged evaluation

Synthetic contracts must cover: exactSOXX F=0/risk equality; positive F but oldJnegative remaining eligible when tail-feasible; both-cost risk gradients when one cost is inactive for F; drawdown initial mark/ties; fractional-tail CVaR values; finite-difference risk adjoints away from ties; full-context short-path gradient parity; signed post-primal λupdates/projection; nonfinite failure; incumbent selection using fixed F/feasibility rather than varying L; ensemble feasibility recomputation; full fees and carried holdings.

Verify identical Phase5initial tensors for each seed and identical input/scaler hashes. Development remains2016–2022, strict-prior annual fitting with at least252mature daily rows; no H5purge, future labels or fold warm-start. Keep the prespecified annual-refit and frozen2023parameter diagnostic modes, seal development selection before those paths, and never choose between modes after viewing results.

All original OOSnet-excess,5/7year stability, median/leave-one-year-out, paired uncertainty, MaxDD/CVaR, same≥2of3seeds-positive-at-both-costs, invalid-event and new-forward requirements stay unchanged. Include realized beta/HAC, costs, turnover and source-sensitivity requirements. Training feasibility is not future risk control or stable alpha.

If the three new training families are authorized and preregistered, campaign family count increases17→20. The prior pure training-checkpoint diagnostic did not add a family. Old histories, seeds and optimization checkpoints are not independent evidence.

Falsification is explicit: if no new trajectory finds positive-F feasible states, retain nulls and report failure in this optimizer/class without claiming mathematical impossibility. If training-feasible states fail OOSgates, risk alignment solved only the training mismatch, not generalization. If an apparent improvement relies on changed sources, hidden objective scaling, different seeds/updates or a relaxed gate, the comparison is invalid. No live execution or promotion follows from this proposal.


登记补充：CVaR边界的相等收益采用平均分配剩余尾部权重，替代草案的时间顺序选择；登记先于拟合，经济风险值不变。
