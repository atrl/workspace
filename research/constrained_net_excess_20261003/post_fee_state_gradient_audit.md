# Post-fee holding-state derivative audit

Durable audit evidence is archived under main Recorder `300530137db54b6c807c39ebb9e8c331` at `analysis/diagnostics/post_fee_state_gradient/`; [full synthetic JSON](qlib_recorder/741902640886742623/300530137db54b6c807c39ebb9e8c331/artifacts/analysis/diagnostics/post_fee_state_gradient/post_fee_state_gradient_audit.json). The temporary paths below identify the original audit execution. This later synthetic finding did not alter the Phase7 models or numerical results.

**Confirmed: the legacy graph can give a cost-dependent holding-state derivative at a no-trade point even though the mathematical state map has a unique cost-independent derivative. Forward accounting remains correct.** This is a synthetic design check, not a model fit or evidence of improved financial performance. No repository source, current Phase7 parameters, or results were changed.

Evidence: `/tmp/post_fee_state_gradient_audit.py`, `.json`, `.log`. Installed Torch2.11.0, float64, one CPU thread. The frozen source `src/model/integrated_allocation.py` remains SHA256 `946f20b41afa528fef19562f696ccda81343a78dbade018d3371e18698d23f30` before and after the audit.

## The identity being checked

Let h be pretrade normalized holdings, z the desired **post-fee** normalized target, c the proportional risky-turnover fee, and n the fraction of initial NAV remaining after funding fees. The exact funding equation is

`n + c * sum(abs(n*z_risky - h_risky)) = 1`.

On a valid root and a simplex target, all opening values equal `n*z`. With asset gross returns a=1+r, therefore

`h_next = z*a / sum(z*a)`.

It does not depend directly on h or c; h can influence it only through the target z(h). For a tangent target direction dz with sum(dz)=0 and G=z·a, its unique directional Jacobian is

`d h_next = dz*a/G - z*a*(dz·a)/G²`.

This smooth state identity remains valid at zero turnover. It is separate from the scalar fee and net-NAV maps, which have kinks.

## Requested S-to-cash example

Use h=z=(0,1,0), z(theta)=(0,1-theta,theta), r=(.03,.01,0), at theta0 with a feasible right perturbation. The correct derivative is `(0,-1/1.01,+1/1.01)` at every cost.

| Cost | Correct cash-weight derivative | Legacy autograd cash derivative | Absolute state error |
|---|---:|---:|---:|
|10bp|0.990099009901|0.991090100001|0.000991090100|
|25bp|0.990099009901|0.992580461054|0.002481451153|

Independent fee-affordable NumPy settlement and one-sided finite differences agree with the analytic state derivative. The legacy result in this case equals `1/[(1-c)*1.01]`, exposing the spurious cost factor.

At no-trade all four sign roots have the same forward n. The legacy cash expression is `1-sum(n*z_risky)-c*abs(trade).sum()`. The selected root supplies a derivative for n, while the elementary absolute-value operation gives its zero subgradient at trade0. These choices need not satisfy the derivative of the funding identity, so the cancellation needed for the unique normalized holding-state derivative is lost. This is a graph-representation issue, not incorrect forward cash debits or an assertion that PyTorch's elementary `abs` rule is itself a bug. PyTorch documents that differentiability of the constituent operations is required for ordinary automatic differentiation and specifies separate rules at kinks: https://docs.pytorch.org/docs/2.14/notes/autograd.html#gradients-for-non-differentiable-functions .

## Broader synthetic controls

The audit covers51 scenarios across0/10/25bp: S-to-cash and S-to-L departures; four tangent directions at mixed h=(.2,.6,.2); perturbing h alone with z fixed; a coupled HOLD target with coefficients.7 and1; explicit `target=None`; and away-from-kink targets. Return-coordinate Jacobians are checked separately.

- Legacy no-trade state-JVP error reaches0.002481451153. With fixed target and perturbed pretrade holdings it can return a nonzero state derivative even though the correct derivative is exactly0.
- A coupled HOLD coefficient.7 retains a smaller but measurable erroneous direct-state term. Pure coupled HOLD z=h and the explicit `None` HOLD branch pass: the finding does **not** mean every HOLD derivative is wrong.
- Away from the kink, the legacy state derivative agrees with the analytic map within4.44e-16. Cost0 and return-coordinate controls also behave correctly.
- The algebraic prototype below matches the analytic holding-state Jacobian across all51 cases within4.44e-16. Independent NumPy finite-difference state checks differ by at most2.34e-10.
- Forward net NAV, fees, turnover, cash-debit identity, funding identity and normalized holdings agree with the old kernel/independent settlement within2.22e-16.

## Minimal prospective graph correction

Keep the existing sign-root formulas, branch selection and nonsmooth derivative of n. For a supplied post-fee target, compute:

```
risky_values = n * z_risky
turnover = sum(abs(risky_values - h_risky))
fees = c * turnover                         # actual reported transaction costs
execution_cash = 1 - sum(risky_values)      # retains the fee reserve
open_values = n * z                         # all three assets
ending_values = open_values * (1 + r)
net_equity = sum(ending_values)
ending_weights = z * (1 + r) / sum(z * (1 + r))
```

Keep explicit forward checks against `execution_cash-fees`, the funding equation, and the independent fee-funded settlement, with the existing tolerances. Invalid/nonpositive roots, nonphysical returns, nonfinite values and invalid simplex inputs still fail closed. Keep the existing explicit no-target HOLD behavior. No fee smoothing, trade band, custom backward, straight-through estimator or detached replacement gradient is needed: the state graph is the exact algebraic state function.

Do not edit the frozen Phase5/7 implementation under its original evidence identity. If adopted, implement a separately versioned kernel/gradient path for the next registered experiment and rerun its synthetic and inference/accounting contracts before any new fit. The optimization trajectory would change, so current stored checkpoints/results must not be relabelled as produced by this graph.

## The scalar-boundary limitation remains

The prototype repairs the unique holding-state Jacobian. It does **not** certify all scalar transaction-cost or net-NAV derivatives at no-trade. For the requested S-to-cash right direction, the retained first sign branch happens to give the correct scalar log-NAV derivative after the algebraic rewrite:

-10bp: prototype -0.01090199110 vs independent right FD -0.01090199108.
-25bp: prototype -0.01240725576 vs independent right FD -0.01240725553.

But S-to-L provides an explicit counterexample: both old and prototype autograd return0.01980198020, whereas the feasible right derivatives are approximately0.01779997832 at10bp and0.01478944898 at25bp. The selected no-trade sign root does not follow that particular buy/sell direction. The reported `fees=c*abs(trade)` graph still gives zero fee derivative at the kink even when a specified right directional derivative is positive.

Every explicitly specified feasible curve has a definite one-sided derivative; the full scalar map is nonsmooth and has no unique ordinary Jacobian there. The report includes left/right finite differences where both directions are feasible. Central scalar secants at such points are diagnostics, not claimed true derivatives. A later objective that directly consumes the reported fee tensor must specify its own consistent kink treatment; this state-only correction supplies no such certificate.

No profitable allocation, lower future risk, global optimizer convergence or deployment readiness follows from this derivative audit. It establishes a narrow mathematical correction suitable for a new pre-fit implementation contract.
