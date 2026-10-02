# Nested sparse experts with a direct holdings skip — pre-fit proposal

Status: registered before implementation and market fits; replacing the earlier30-input-hidden/284parameter draft **before registration, implementation or market fitting**. No new policy, forecast, portfolio path, threshold or optimization outcome was evaluated to choose this revision. Proposed new family count: **20→23**. Existing Phase7 results and failed historical qualification are unchanged.

## Evidence and scope

Phase7 Recorder `300530137db54b6c807c39ebb9e8c331` found positive-F, four-tail-feasible models in all42 conditional development seed fits and14 ensembles, but did not establish stable excess. Saved-book fee diagnostic `8fdcc6b190814dd499a4cd2802e2aaa9` attributes price_vol's25bp development annualized log excess to +2.5445% gross position contribution and −2.0823% fees, leaving +0.4623%; six gross-positive years became two net-positive years. The2023+ reused holdings already have negative gross excess. Trading cost is a material problem, not a complete explanation of recent underperformance.

The old market-only expert gate has `target=q(x)+alpha_H(x)*h`; for fixed x and alpha_H<1, exactHOLD occurs at a single h. Its globaltheta budget also prevents fullSOXL in one state and fullcash in another from identicalSOXX holdings. The new hypothesis is a **nested structural extension** that removes this cross-state budget coupling and allows learned holding-dependent no-order regions. No manual band, timer, smoothing, extra objective or market information is added.

Prefer the minimal extension: keep the exact parent27→8sigmoid market encoder, replace globaltheta with an unconstrained four-expert score head, and add a direct linear4×3holdings skip before sparsemax. This adds21parameters, not33, and can reproduce the parent's entire policy class and every initial trajectory. It jointly changes the action parameterization and its holdings dependence; it is not a clean causal test of sparsemax alone. The separately proved settlement state-derivative correction and scalar kink convention are also registered before fits; economic changes cannot be attributed to architecture alone.

## Frozen inputs and controls

Use the same qualified native price source, as_of2026-09-25, original25market features and existing annual OOFvolatility prediction/availability. Reuse hash-verified Phase7/Phase5 inputs, forecast objects, nativeDatasetH, mature-date membership and prior-only27channel scalers. No forecast retraining or source refresh. Holdings are current, self-owned, post-return/pretrade normalized state, not an extra stored feature estimated from a benchmark or another model.

| New family | Market27 mask | Holdings skip |
|---|---|---|
| holdings_only | all channels exactly0 after the original scaler | active |
| price_holdings | originalprice25; volatility/mask0 | active |
| price_vol_holdings | originalprice25 + savedvol + availability | active |

All variants have27market inputs,3state inputs through the direct skip,8sigmoidunits and4expert scores. Their nominal architecture is identical. Map the masks explicitly to oldconstant/price/price_vol preparation, retaining existing immutable scaler/date/matrix identities. Unusedvolerrors cannot contaminate the first two controls; active-invalid training/replayvol follows the same fail-closed contracts.

`holdings_only` is a state-only control, not a constant/no-information control: h reflects earlier returns, past decisions and the execution-open mark. Its market encoder sees0. Under identical zero biases, symmetric parenttheta embedding, K=0 and deterministic training, the three seed-dependent encoderweight tensors are unused and may give identical decision/optimization trajectories. Disclose that symmetry and measure equality; those seeds never become independent market evidence. Conditional variants still test the original three fixed encoder seeds.

Training remains all2011+ rows with feature and one-day outcome availability strictly before each evaluation-year January1NewYork, minimum252mature rows, fresh annual initialization and no warm-start. Development2016–2022; selected diagnostic2023+ annual/frozen2023 modes only after the development seal. Keep cachedforecast rules, same Aug3 actual-trade window, fee/cash assumptions, source sensitivity and fresh-forward requirements.

## Architecture:272parameters and explicit parent embedding

Let v(x)=sigmoid(Wx+b), with the **parent27→8encoder unchanged**. Define four scores in order[L,S,C,H]:

`u(x)=A*v(x)+a`,

`ell(x,h)=u(x)+K*h`, with K of shape4×3,

`q=sparsemax(ell)`,

`target=q_L*e_L+q_S*e_S+q_C*e_C+q_H*h`.

The new parameter count is `(27*8+8)+(4*8+4)+(4*3)=224+36+12=272`, versus parent's `224+27=251`. No nonlinear holding encoder is needed for the proposed HOLD-region property. This is a modest capacity increase and a changed optimizer geometry, not a parameter-matched experiment.

For **any** legal parenttheta with three rows[L,C,H] and ninecolumns[constant,8hidden], set K=0 and embed its expert scores exactly:

- A_L=theta_L,hidden; a_L=theta_L,constant.
- A_C=theta_C,hidden; a_C=theta_C,constant.
- A_H=theta_H,hidden; a_H=theta_H,constant.
- A_S=−(A_L+A_C+A_H); a_S=1−(a_L+a_C+a_H).

The parent has nonnegativealphas with sum≤1, hence these four scores already form a legal simplex: `[alpha_L,1−sum(alpha),alpha_C,alpha_H]`. Sparsemax is identity on that simplex, and the resulting target is the parenttarget for every x and h. Therefore the whole original class is contained in the new class; this is a formal inclusion claim, not a claim that the optimizer will retain or improve any particular parent model.

After initialization A,a,K are unconstrained parameters. There is no cross-state sumtheta budget. Sparsemax ensures the action weights remain a simplex on every forward pass.

## Initialization without a starting-policy confound

For each fresh fold/seed17/41/71, initialize the encoder by the **exact parent constructor**: same27inputdimension, PyTorchfloat64 weight convention, RNG scope/order and zero hiddenbias. Start from a fresh parenttheta=1/54 and apply the above embedding; K=0. Do not load a fitted parent allocation model.

All initial targets, realized owned holdings, fees, risks and F must match the fresh Phase7parent for arbitrary test x/h and for synthetic complete paths at bothcosts, within declared floating-point parity. Compare the encoder tensors bit-for-bit. This removes the earlier draft's uniform-four-expert versus parent initial-risk confound. No uniformhead, extra start, hiddenepsilon or newseed is permitted.

Initialscores are interior under finite parentfeatures. The directK→score→q derivative is structurally active immediately; unlike a zero outputhead, the initialized parenthead also does not automatically block all encoder gradients. This does not guarantee a nonzero empirical objectivegradient on every datum.

Keep an independent exactSOXX incumbent: A=0, a=[0,1,0,0], K=0, giving q=[0,1,0,0] for all finite states. It is separately audited and is not the only optimization start. Nullselection is legitimate; numericalfallback is disqualifying.

Primal updates become ordinaryAdam on W,b,A,a,K: lr0.003, no weight decay, globalclip5,120updates. The oldglobaltheta parameterprojection is gone; simplex projection is now in the **forward activation**, and the signeddual nonnegativeprojection is unchanged. Even with identical initialpaths, new parameter geometry can alter finite-step learning; disclose this.

## Sparsemax, exact HOLD and finite witnesses

Sparsemax projects onto `{q>=0,sum(q)=1}`, not the oldL1ball with sum≤1. Determine the sorted support/thresholdtau and use q_i=max(ell_i−tau,0). On strictpositive supportI use Jacobian `diag(1_I)−1_I*1_I.T/|I|`. At exact zero-boundary coordinates use that stated support convention; no supportepsilon or straight-through gradient. This is the projection/derivative mechanism of [Martins and Astudillo,2016](https://proceedings.mlr.press/v48/martins16.html), not evidence of trading alpha. Retain the financialFobjective; do not import the paper's classification loss.

Use a common logit offset for numerical stability, finite checks and simplex verification. Single-support output is mathematically an exactonehot. No temperature, logitclipping, entropybonus or numerical “almostHOLD” rule is added. Numerical invalidity is a recorded failure, not a reason to alter the activation after results.

At fixedmarketx, exactHOLD has a particularly clear geometry:

`q_H=1 iff (u_H−u_j)+(K_H−K_j)h >=1 for every j≠H`.

These linearinequalities intersect the holdings simplex in a learned polyhedral region, possibly empty or the whole simplex. They are generated by trainable scores, not a user-entered tradeband. On soleH support, target is an exact copy of h, using the same no-order accounting; h still earns its actual return and remains differentiable through the state update. Record exactHsupport and actual orders separately: a pureS action can also be no-order when S is already owned.

Pre-fit expressivity witnesses are unit-test constructions only, never marketinitializations or candidate rules:

1. One hiddenunit v=sigmoid(u), K=0, scores `[12v−8,0,4−12v,2]`. At u=−log23 the output is exactCASH; at +log23 exactSOXL; for v∈[1/4,3/4] it is exactHOLD. These finite actions occur from the sameSOXX h, proving the old cross-state exclusion is gone.
2. At fixedmarket use u_H=2, others0, K_L=(4,0,0), K_C=(0,0,4), otherKrows0. The region h_L≤1/4 and h_C≤1/4 is exactHOLD with a nonempty interior in the holdingssimplex. This verifies holding dependence without a nonlinearholdingencoder.
3. Random feasibleparenttheta/finiteencoder/x/h embedding must reproduce originaltargets, including theta0, mixed/HOLD boundaries and fullcostpaths. Randomtestvectors are syntheticverification, not a model family or economic parametersearch.

## State timing, exact accounting and conditional derivative repair

h is the previous owned position marked at the current executionopen, before the newtrade/fee. Its direct use in scores exposes contemporaneous position drift as information. Historicalsim assumes observing that mark and executing instantaneously at the same quotedopen; this is an idealfill assumption, not proof an auctionprice was known before an executable order. Futurepaper/runtime work must use genuinely available quote and filltimes. No next-session return or endingholding enters the currentdecision.

The policy remains costblind. Post-fee normalizedtargets and proportionalfees still permit equal normalizedh/q/targetpaths across costs in exact arithmetic; separate settlement remains necessary for NAV, tradednotional, F and risk. Do not claim K learns the costrate from h.

The legacy no-trade state-Jacobian defect is confirmed by the separate post-fit audit: some partial/departure derivatives spuriously depend on the fee rate, although forward money values and the pure coupled HOLD composition are correct. Old sources and their model evidence remain unchanged.

This study implements `src/model/post_fee_settlement.py`. It retains the exact funding equation and computes normalized ending holdings directly from the target. Its scalar funding backward uses the registered implicit sign-zero convention: with actual risky trade signs s (exact zeros map to0), D=1+c*s.dot(z), dn/dz=-c*n*s/D and dn/dh=c*s/D. Actual fees/turnover are still calculated from exact funded orders; no smoothing or tolerance-based trade band is introduced.

The independent [pre-fit mathematical audit](implicit_settlement_gradient_audit.md) verifies this selection is a convex combination of adjacent ambient funding-map gradients when denominators are positive. It is not every feasible one-sided derivative and does not establish a blanket Clarke-gradient claim for arbitrary composed networks. The normalized-state derivative itself is unique and smooth on valid positive-NAV targets. The audit checks independent forward cash accounting, smooth-region finite differences, pure HOLD, partial zero trades and the explicit one-sided limitation. Complete numerical results are summarized in [settlement_audit.json](settlement_audit.json).

This is therefore a combined nested-architecture and derivative-correctness intervention, not architecture-only causal attribution. The prospective JSON registration remains unchanged. No market fit is permitted before implementation, synthetic information, data and source-seal checks all pass.

## Same full-pathF, risks, dualprotocol and selection

At10/25bp use each own fullchronological CASH-start path, all entry/subsequentfees, zerocashinterest and no terminal liquidation. Keep `A_c=63*mean(log g_policy−log g_same_cost_SOXX)`, `F=min(A10,A25)`;63 remains the inherited unit scale, not a selected holdinghorizon. Oldquarter-RMSJ is report-only.

Keep the samefour anchoredMaxDD/exactfractionalCVaR5 constraints, positive fixed same-trainingbenchmarknormalizers,1e−10percentagepoint numericalfeasibility allowance, equal exactboundaryCVaRmass and earliestDDpeak/trough. Lambda starts0; after each primalproposal/fullreevaluation use signed `lambda_next=max(0,lambda_used+.003*g_post)` withoutcap. Bothcostriskgradients contribute even for an inactiveFbranch; exactworst-Fties split evenly.

Separate currentlearner/evaluation from exactSOXX/best feasibleincumbent. Evaluate epoch0plus120updates; retain the first strictlyhigherF only if allfour rawrisks pass. Never rank by changingLagrangian. All3seedincumbents remain; their equal-target ensemble owns one book, and each component uses the **same ensembleh** before targets/q are averaged. Audit its ownfullF/riskpath at bothcosts; F<0 orriskfailure selects exactSOXX, never a seedsubset. Anyseed/ensemble numericalfailure propagates to qualification separately from validnulls.

Keep all existing annual5/7, median/leave-one-year-out, family-adjusted20/60bootstrap, same≥2/3seed profit at bothcosts, tail, invalid-event, sourcesensitivity and newforward requirements. Everyfield's originalunits/tolerances are preserved. Increase registeredfamilycount to23. Reducedtradefrequency, nonemptyHOLDregions or trainingfeasibility alone are not promotiongates.

## Minimal chronological implementation and tests

For NumPy fullpath inference, precompute currentparameter marketlogits u(X) of shapeN×4. At eachdate compute ell=u_t+K*h_t, sparsemax, exacttarget and existingfee-fundedstateupdate. Holdings-conditionedq cannot be cached from a stale/benchmark/teacher-forcedh.

For Torchgradients, precompute u(X) **with its graph attached**, gather into63scored+up-to21context batches, and sequentially compute K*h,q,target,settlement across at-most84steps. Initialize each contextboundary with detached trueh from the current fullforwardpath. No detach inside context/scoredstate recurrence; K→h→futureq effects must remain differentiable within the chosen truncation. FullF/riskadjoints are computed globally, and contextrows transmit derivatives without double-scoring. Allfees remain in the fullforwardobjective.

Check NumPy/Torchsupport,q,targets,fees,netmultipliers and endingh parity, not only returns. Away from kinks, compare shortfullcontext gradients with a completechronologicalautograd/finite-difference reference, includingK and bothcostriskterms. Preserve explicit21contextapproximation; exactglobalrisks do not imply untruncatedstategradients.

Before any marketfit: sparsemax/Jacobianboundarytests; parentclass/initialtrajectoryembedding; exactS/H/L/C accounting; holdingregionwitness; mask/causalstatechecks; incumbent/ensemble/failureprotocol; signedpost-primaldual; and a fixedknown-information syntheticchronology with an independently reachable positiveF/tailfeasiblepolicy. Test declared120updates/3seeds on that toy, without a horizon/temperature/startgrid. If it cannot use known availableinformation, resolve technicalinterpretation before marketfits; never rescue it after seeing marketresults.

Profile only syntheticrepresentativeN with CPUfloat64/oneTorchthread and the registeredprocesslayout. Use sequentialNumPy plus batchedTorch84steprecurrence; the directskip avoids an extra nonlinearholdingnetwork. Timingtests may guide neutral implementation efficiency, not epochs, starts or financial parameters.

## Diagnostics, reuse and falsification

Sparse singlevertices have zero logitJacobian. AllHOLD/allSOXX collapse can become locally stuck even though alternatives are representable. Archive121states withF/A,oldJ,raw/normalizedrisks,lambda_used/postg/lambda_next,incumbentidentity,gradient/primalstepnorms and supportcardinality1/2/3/4fractions. Add exactsoleL/S/C/H fractions, supportchanges, logitscale, encoder/head/Kgradientnorms and exactzerogradientevents. Save selectedq/h/targets/fees/fullpaths in Recorder. Report state-onlyseedpath equality and actualtrades separately from support, including exactzeroordercounts versus the inherited reportingtolerance.

No postmarketentropy, softmaxwarmup, epsilon, logitnoise, restarts, parametercaps, additionalupdates or gatechanges. Representabilitytests prove classcapacity; toyfits prove a narrow optimizationmechanism; retainedtrainingpaths prove historicaltrainingfeasibility. None proves stablefutureexcess or a globaloptimum. Failure may reflect optimization, insufficientinformation or generalization; do not infer whole-class impossibility from one registeredtrajectory.

Reuse `IntegratedAllocationPolicy`'s ownedstatecontract; parentfee/accounting kernels and independentchecks; Phase7F/tailadjoints/dual/incumbent/ensembleprotocol; and frozenDatasetH/scalers. Oldholdings-awarePIT trade reductions relied on EWM/threshold/approximatefees and failedriskgates; they are not evidence for this learnedregion. Existing`project_theta` is anL1ballparameterprojection, not a replacement for a four-score simplexactivation. No existing sparsemax/entmax implementation was found in src/tests.

Implementation belongs in newmodules plus an executable registerednotebook and QlibRecorder. Keep oldsource/providers/cards immutable, and record any confirmed derivativefix distinctly. Largepaths/checkpoints remain Recorder-only; the card stays small with source/planhashes, selected/gate/annual/window summaries and Chineseconclusion, following the [official Qlibworkflow contract](https://qlib.readthedocs.io/en/latest/component/workflow.html). No broker/SIM/deploymentchange is authorized by this proposal alone.

