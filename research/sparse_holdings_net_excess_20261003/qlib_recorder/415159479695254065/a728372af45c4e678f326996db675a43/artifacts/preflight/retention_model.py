"""Sparse four-expert actions with direct, actually owned holdings input.

The parent market encoder and fresh initial policy embed exactly. Forward
simplex projection replaces the old global parameter budget. All financial
objectives/constraints remain in their original units; no trading band is used.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.model.benchmark_anchor_utility import (
    BenchmarkAnchorPolicy, NumericalFitFailure, _matrix, _weights, _audit_anchor_paths,
    policy_to_dict as parent_policy_to_dict, risk_groups, utility_and_daily_weights,
)
from src.model.constrained_net_excess import (
    ConstrainedNetExcessSpec, _parent_spec, _risk_record, _fallback_evaluation,
    feasible_improvement, gradient_proxy, update_multipliers,
)
from src.model.integrated_allocation import _cpu_threads
from src.model.post_fee_settlement import settle_torch, scalar_settle
from src.objective.benchmark_tail_constraints import TailConstraintError, evaluate_constraints, reference_risk

EXPERTS = ("soxl", "soxx", "cash", "hold")
SPARSEMAX_REFERENCE = "https://proceedings.mlr.press/v48/martins16.html"


class _Sparsemax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        if scores.shape[-1:] != (4,) or not bool(torch.isfinite(scores).all()):
            raise NumericalFitFailure("invalid four-expert scores")
        shifted = scores - scores.max(-1, keepdim=True).values
        if not bool(torch.isfinite(shifted).all()):
            raise NumericalFitFailure("nonfinite centered sparsemax scores")
        ordered = shifted.sort(-1, descending=True).values
        cumulative = ordered.cumsum(-1)
        ranks = torch.arange(1, 5, dtype=scores.dtype, device=scores.device)
        selected = 1 + ranks * ordered > cumulative
        count = selected.sum(-1, keepdim=True)
        threshold = (cumulative.gather(-1, count - 1) - 1) / count
        q = (shifted - threshold).clamp_min(0)
        support = q > 0
        if bool((support.sum(-1) == 0).any()) or bool(((q.sum(-1) - 1).abs() > 1e-12).any()):
            raise NumericalFitFailure("invalid sparsemax simplex")
        ctx.save_for_backward(support)
        return q

    @staticmethod
    def backward(ctx, gradient):
        (support,) = ctx.saved_tensors
        masked = gradient * support
        mean = masked.sum(-1, keepdim=True) / support.sum(-1, keepdim=True)
        return (gradient - mean) * support


def sparsemax(scores: torch.Tensor):
    """Projection onto sum-one simplex; exact zero coordinates are inactive."""
    return _Sparsemax.apply(scores)


def sparsemax_numpy(scores):
    value = np.asarray(scores, dtype=np.float64)
    if value.shape[-1:] != (4,) or not np.isfinite(value).all():
        raise NumericalFitFailure("invalid numpy four-expert scores")
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = value - value.max(axis=-1, keepdims=True)
    if not np.isfinite(shifted).all():
        raise NumericalFitFailure("nonfinite centered numpy scores")
    ordered = np.sort(shifted, axis=-1)[..., ::-1]
    cumulative = np.cumsum(ordered, axis=-1)
    count = np.sum(1 + np.arange(1, 5) * ordered > cumulative, axis=-1, keepdims=True)
    threshold = (np.take_along_axis(cumulative, count - 1, axis=-1) - 1) / count
    q = np.maximum(shifted - threshold, 0.)
    if not np.isfinite(q).all() or np.any(np.abs(q.sum(-1) - 1) > 1e-12):
        raise NumericalFitFailure("invalid numpy sparsemax simplex")
    return q


def _torch_target(q, held):
    target = q[..., :3] + q[..., 3:] * held
    sole_hold = (q[..., 3] == 1) & ((q > 0).sum(-1) == 1)
    return torch.where(sole_hold[..., None], held, target)


class SparseHoldingsPolicy(nn.Module):
    feature_count = 27
    hidden_units = 8

    def __init__(self, *, seed=17, anchor=False):
        super().__init__()
        parent = BenchmarkAnchorPolicy(seed=seed)
        self.hidden = deepcopy(parent.hidden)
        self.A = nn.Parameter(torch.zeros((4, 8), dtype=torch.float64))
        self.a = nn.Parameter(torch.zeros(4, dtype=torch.float64))
        self.K = nn.Parameter(torch.zeros((4, 3), dtype=torch.float64))
        self.seed = int(seed)
        self.parent_initial_tensor_sha256 = parent_policy_to_dict(parent)["tensor_sha256"]
        with torch.no_grad():
            if anchor:
                self.a[1] = 1.
            else:
                self._copy_parent_head(parent.theta)

    def _copy_parent_head(self, theta):
        self.A[[0, 2, 3]] = theta[:, 1:]
        self.a[[0, 2, 3]] = theta[:, 0]
        self.A[1] = -theta[:, 1:].sum(0)
        self.a[1] = 1 - theta[:, 0].sum()
        self.K.zero_()

    def market_scores(self, features):
        if features.shape[-1:] != (27,) or not bool(torch.isfinite(features).all()):
            raise NumericalFitFailure("invalid market feature tensor")
        if is_exact_anchor(self):
            # Scores are still projected by the generic sparsemax, so the
            # declared singleton Jacobian (including a/K) remains exactly zero.
            link = sum((p * 0).sum() for p in self.parameters())
            return self.a + link + (features * 0).sum(-1, keepdim=True)
        hidden = self.hidden(features)
        if not bool(torch.isfinite(hidden).all()):
            raise NumericalFitFailure("nonfinite market preactivation")
        scores = torch.sigmoid(hidden) @ self.A.T + self.a
        if not bool(torch.isfinite(scores).all()):
            raise NumericalFitFailure("nonfinite market expert scores")
        return scores

    def experts_from_market_scores(self, market_scores, holdings):
        return sparsemax(market_scores + holdings @ self.K.T)

    def expert_weights(self, features, holdings):
        return self.experts_from_market_scores(self.market_scores(features), holdings)

    def forward(self, features, holdings):
        return _torch_target(self.expert_weights(features, holdings), holdings)

    def predict_target(self, features, pretrade_weights):
        return policy_target(self, features, pretrade_weights)


def embed_parent_policy(parent):
    """Class-inclusion construction for verification; fitter never loads a fitted parent."""
    parent_policy_to_dict(parent)  # Validate the original legal parameter domain.
    model = SparseHoldingsPolicy(seed=parent.seed)
    model.hidden.load_state_dict(parent.hidden.state_dict())
    with torch.no_grad():
        model._copy_parent_head(parent.theta)
    return model


def _validate_model(model):
    if not isinstance(model, SparseHoldingsPolicy) or any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise NumericalFitFailure("nonfinite or incompatible sparse-holdings model")


def policy_to_dict(model):
    _validate_model(model)
    state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode()); digest.update(str(state[name].shape).encode()); digest.update(state[name].tobytes())
    return {"schema_version": "sparse_holdings_v1", "feature_count": 27, "hidden_units": 8,
            "parameter_count": 272, "dtype": "float64", "seed": model.seed,
            "tensor_sha256": digest.hexdigest(), "state_dict": state}


def policy_from_dict(payload):
    if (payload.get("schema_version") != "sparse_holdings_v1" or payload.get("dtype") != "float64"
            or payload.get("feature_count") != 27 or payload.get("hidden_units") != 8 or payload.get("parameter_count") != 272):
        raise ValueError("blocked: incompatible sparse-holdings policy schema")
    model = SparseHoldingsPolicy(seed=int(payload["seed"]), anchor=True)
    model.load_state_dict({name: torch.as_tensor(value, dtype=torch.float64) for name, value in payload["state_dict"].items()})
    _validate_model(model)
    if policy_to_dict(model)["tensor_sha256"] != payload.get("tensor_sha256"):
        raise ValueError("blocked: sparse-holdings tensor hash differs")
    return model.eval()


def is_exact_anchor(model):
    model = policy_from_dict(model) if isinstance(model, dict) else model
    _validate_model(model)
    return bool(torch.count_nonzero(model.A) == 0 and torch.count_nonzero(model.K) == 0
                and torch.equal(model.a.detach(), model.a.new_tensor([0., 1., 0., 0.])))


def zero_anchor(seed=17):
    return SparseHoldingsPolicy(seed=seed, anchor=True)


def _numpy_market(model, x):
    _validate_model(model)
    p = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    if is_exact_anchor(model):
        return np.tile([0., 1., 0., 0.], (len(x), 1)), p["K"]
    z = x @ p["hidden.weight"].T + p["hidden.bias"]
    if not np.isfinite(z).all():
        raise NumericalFitFailure("nonfinite numpy market preactivation")
    gate = np.empty_like(z); positive = z >= 0
    gate[positive] = 1 / (1 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive]); gate[~positive] = exp_z / (1 + exp_z)
    market = gate @ p["A"].T + p["a"]
    if not np.isfinite(market).all():
        raise NumericalFitFailure("nonfinite numpy market scores")
    return market, p["K"]


def _numpy_decision(markets, skips, step, held):
    experts, targets, sole, scale = [], [], [], 0.
    for market, skip in zip(markets, skips):
        logits = market[step] + skip @ held
        q = sparsemax_numpy(logits)
        hold = bool(q[3] == 1 and np.count_nonzero(q) == 1)
        target = held.copy() if hold else q[:3] + q[3] * held
        experts.append(q); targets.append(target); sole.append(hold)
        scale = max(scale, float(np.max(np.abs(logits))))
    exact_hold = all(sole)
    target = held.copy() if exact_hold else np.mean(targets, axis=0)
    return np.mean(experts, axis=0), target, exact_hold, scale


def policy_target(models, features, pretrade_weights):
    policies = list(models) if isinstance(models, (tuple, list)) else [models]
    if not policies: raise ValueError("blocked: empty sparse ensemble")
    x = _matrix(np.asarray(features).reshape(1, -1), 27, "features")
    prepared = [_numpy_market(model, x) for model in policies]
    return _numpy_decision([p[0] for p in prepared], [p[1] for p in prepared], 0, _weights(pretrade_weights))[1]


def numpy_rollout(models, features, returns, *, cost_bps, initial_weights=None, force_soxx_mask=None):
    x, r = _matrix(features, 27, "features"), _matrix(returns, 2, "returns")
    if len(x) != len(r) or not len(x) or (r <= -1).any() or not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("blocked: invalid sparse chronological inputs/cost")
    policies = list(models) if isinstance(models, (tuple, list)) else [models]
    if not policies: raise ValueError("blocked: empty sparse ensemble")
    force = np.zeros(len(x), dtype=bool) if force_soxx_mask is None else np.asarray(force_soxx_mask)
    if force.dtype != np.bool_ or force.shape != (len(x),): raise ValueError("blocked: explicit per-date SOXX mask required")
    prepared = [_numpy_market(model, x) for model in policies]
    markets, skips = [p[0] for p in prepared], [p[1] for p in prepared]
    held = _weights([0., 0., 1.] if initial_weights is None else initial_weights).copy()
    n = len(x)
    result = {name: np.empty((n, 3)) for name in ["target_weights", "pretrade_weights", "ending_weights", "open_net_weights", "execution_target_weights"]}
    result.update({name: np.empty(n) for name in ["net_equity", "fees", "turnover", "open_net_equity", "logit_scale"]})
    result["expert_weights"] = np.empty((n, 4)); result["exact_hold"] = np.empty(n, dtype=bool)
    for i in range(n):
        q, target, hold, scale = _numpy_decision(markets, skips, i, held)
        if force[i]: q, target, hold = np.array([0., 1., 0., 0.]), np.array([0., 1., 0.]), False
        result["pretrade_weights"][i] = held; result["target_weights"][i] = target
        ending, net, fee, turnover, executable, opened = scalar_settle(*held, *target, *r[i], cost_bps / 10000.)
        result["ending_weights"][i] = ending; result["open_net_weights"][i] = target
        result["execution_target_weights"][i] = executable
        result["net_equity"][i], result["fees"][i], result["turnover"][i], result["open_net_equity"][i] = net, fee, turnover, sum(opened)
        result["expert_weights"][i], result["exact_hold"][i], result["logit_scale"][i] = q, hold, scale
        held = np.asarray(ending)
    result["net_return"] = result["net_equity"] - 1; result["equity"] = result["net_equity"].cumprod()
    result["exact_zero_order"] = result["turnover"] == 0
    if not all(np.isfinite(value).all() for value in result.values()) or (result["equity"] <= 0).any():
        raise NumericalFitFailure("nonfinite sparse owned path")
    return result


def replay_policy(models, features, returns, *, cost_bps, initial_weights=None, force_soxx_mask=None):
    result = numpy_rollout(models, features, returns, cost_bps=cost_bps, initial_weights=initial_weights, force_soxx_mask=force_soxx_mask)
    frame = pd.DataFrame({"strategy_return": result["net_return"], "equity": result["equity"], "turnover": result["turnover"],
        "fees": result["fees"], "cost_bps": cost_bps, "execute_open_target": ~result["exact_hold"],
        "execute_open_trade": result["turnover"] > 1e-12, "exact_hold_support": result["exact_hold"], "exact_zero_order": result["exact_zero_order"]})
    for prefix, name in [("", "open_net_weights"), ("pretrade_", "pretrade_weights"), ("ending_", "ending_weights"), ("execution_", "execution_target_weights")]:
        for j, asset in enumerate(EXPERTS[:3]): frame[prefix + asset + "_weight"] = result[name][:, j]
    for j, expert in enumerate(EXPERTS): frame["expert_" + expert] = result["expert_weights"][:, j]
    return frame


def _hash_array(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def _path_diagnostics(path):
    q = path["expert_weights"]; support = q > 0; count = support.sum(1)
    return {"support_cardinality_fraction": {k: float(np.mean(count == k)) for k in (1, 2, 3, 4)},
            "sole_expert_fraction": {name: float(np.mean((q[:, i] == 1) & (count == 1))) for i, name in enumerate(EXPERTS)},
            "support_change_count": int(np.any(support[1:] != support[:-1], axis=1).sum()),
            "exact_hold_count": int(path["exact_hold"].sum()),
            "exact_zero_order_count": int((path["turnover"] == 0).sum()),
            "reported_trade_count_gt_1e_minus12": int((path["turnover"] > 1e-12).sum()),
            "maximum_absolute_logit": float(path["logit_scale"].max())}


def _evaluate(models, x, returns, *, spec, anchors, references):
    paths, components, constraints, legacy = {}, {}, {}, {}
    mean_spec, old_spec = _parent_spec(spec, downside=0), _parent_spec(spec, downside=1)
    for cost in spec.cost_bps:
        path = numpy_rollout(models, x, returns, cost_bps=cost)
        logs = np.log(path["net_equity"]); relative = logs - np.log(anchors[cost]["net_equity"])
        paths[cost] = path; components[cost] = utility_and_daily_weights(relative, spec=mean_spec)
        constraints[cost] = evaluate_constraints(logs, references[cost], risk_tolerance_percentage_points=spec.risk_tolerance_percentage_points)
        legacy[cost] = utility_and_daily_weights(relative, spec=old_spec)["utility"] + math.sqrt(spec.rms_epsilon)
    A = {c: components[c]["utility"] for c in spec.cost_bps}; F = min(A.values())
    active = [c for c, value in A.items() if value == F]
    cost_weights = {c: 1 / len(active) if c in active else 0. for c in spec.cost_bps}
    original = {"J": F, "centered_utility": A, "cost_weights": cost_weights, "components": components, "paths": paths, "anchor_utility": 0.}
    return {"F": F, "A_by_cost": A, "cost_weights": cost_weights, "paths": paths,
            "daily_mean_log_excess_by_cost": {c: components[c]["total_log_excess"] / len(x) for c in spec.cost_bps},
            "aggregate_log_excess_by_cost": {c: components[c]["total_log_excess"] for c in spec.cost_bps},
            "legacy_J": min(legacy.values()), "legacy_centered_utility_by_cost": legacy,
            "constraints_by_cost": constraints,
            "normalized_g": np.concatenate([constraints[c]["normalized_g"] for c in spec.cost_bps]),
            "feasible": all(constraints[c]["feasible"] for c in spec.cost_bps),
            "objective_evaluation": original}


def evaluate_policy(models, features, returns, *, spec: ConstrainedNetExcessSpec, anchors=None, references=None):
    x, r = _matrix(features, 27, "features"), _matrix(returns, 2, "returns")
    if len(x) != len(r) or not len(x) or (r <= -1).any(): raise ValueError("blocked: invalid chronological panel")
    anchors = anchors or {c: numpy_rollout(zero_anchor(), x, r, cost_bps=c) for c in spec.cost_bps}
    references = references or {c: reference_risk(np.log(anchors[c]["net_equity"])) for c in spec.cost_bps}
    return _evaluate(models, x, r, spec=spec, anchors=anchors, references=references)


def truncated_utility_gradient(model, features, returns, evaluation, *, spec):
    """Attached market scores plus sequential K*h and true owned-state context."""
    x, r = _matrix(features, 27, "features"), _matrix(returns, 2, "returns")
    if len(x) != len(r) or (r <= -1).any(): raise ValueError("blocked: invalid gradient inputs")
    n = len(x)
    boundaries = [(max(0, start-spec.context_sessions), start, min(n, start+spec.gradient_chunk_length))
                  for start in range(0, n, spec.gradient_chunk_length)]
    longest = max(end-context for context, start, end in boundaries)
    index = np.zeros((len(boundaries), longest), dtype=np.int64)
    valid = np.zeros_like(index, dtype=bool); scored = np.zeros_like(index, dtype=bool)
    for row, (context, start, end) in enumerate(boundaries):
        size = end-context; index[row, :size] = np.arange(context, end)
        valid[row, :size] = True; scored[row, start-context:size] = True
    model.zero_grad(set_to_none=True)
    market = model.market_scores(torch.from_numpy(x))  # Keep W/b/A/a graph attached.
    market_chunks = market[index]
    ret = torch.cat((torch.from_numpy(r), torch.zeros((n, 1), dtype=torch.float64)), dim=-1)[index]
    mask = torch.from_numpy(valid)
    loss = market.sum()*0.
    parity = {k: 0. for k in ("expert_weights", "target_weights", "fees", "net_equity", "ending_weights")}
    support_mismatches = 0
    for cost in spec.cost_bps:
        full = evaluation["paths"][cost]
        held = torch.from_numpy(np.array([full["pretrade_weights"][context] for context, _, _ in boundaries]))
        adjoints = evaluation["components"][cost]["daily_weights"] * evaluation["cost_weights"][cost]
        weights = torch.from_numpy(adjoints[index] * scored)
        for t in range(longest):
            active = mask[:, t]
            q = model.experts_from_market_scores(market_chunks[:, t], held)
            q = torch.where(active[:, None], q, q.new_tensor([0., 0., 0., 1.]))
            target = _torch_target(q, held)
            returns_t = torch.where(active[:, None], ret[:, t], torch.zeros_like(ret[:, t]))
            result = settle_torch(held, target, returns_t, cost/10000.)
            net = torch.where(active, result["net_equity"], torch.ones_like(result["net_equity"]))
            actual = {"expert_weights": q, "target_weights": target, "fees": result["fees"],
                      "net_equity": net, "ending_weights": result["ending_weights"]}
            positions = index[:, t][valid[:, t]]
            for field, value in actual.items():
                observed = value.detach().numpy()[valid[:, t]]
                expected = full[field][positions]
                parity[field] = max(parity[field], float(np.max(np.abs(observed-expected), initial=0)))
                if field == "expert_weights": support_mismatches += int(np.count_nonzero((observed > 0) != (expected > 0)))
            loss = loss - (weights[:, t]*torch.log(net)).sum()
            held = torch.where(active[:, None], result["ending_weights"], held)  # No inner-state detach.
    if max(parity.values()) > 1e-10 or support_mismatches:
        raise NumericalFitFailure(f"sparse Torch/NumPy path mismatch: {parity}, support={support_mismatches}")
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise NumericalFitFailure("missing/nonfinite sparse model gradient")
    norm = lambda parameters: float(torch.sqrt(sum(p.grad.square().sum() for p in parameters)))
    return {"gradient_proxy_value": float(loss.detach()), "forward_parity_max_error": max(parity.values()),
            "forward_parity_by_field": parity, "support_mismatch_count": support_mismatches,
            "scored_rows": int(scored.sum()), "chunk_boundaries": boundaries,
            "encoder_gradient_norm": norm(model.hidden.parameters()), "head_gradient_norm": norm([model.A, model.a]),
            "K_gradient_norm": norm([model.K]), "all_parameter_gradients_exact_zero": all(bool(torch.count_nonzero(p.grad) == 0) for p in model.parameters())}


def _initial_parent_parity(model, x, returns, *, spec, initial):
    from src.model.benchmark_anchor_utility import numpy_rollout as parent_rollout
    parent = BenchmarkAnchorPolicy(seed=model.seed)
    audit = {"parent_initial_tensor_sha256": parent_policy_to_dict(parent)["tensor_sha256"], "costs": {}}
    for name in ("weight", "bias"):
        if not torch.equal(getattr(model.hidden, name), getattr(parent.hidden, name)):
            raise NumericalFitFailure("fresh parent hidden tensors differ")
    Fs = []
    for cost in spec.cost_bps:
        old = parent_rollout(parent, x, returns, cost_bps=cost)
        old_anchor = parent_rollout(BenchmarkAnchorPolicy(seed=model.seed, anchor=True), x, returns, cost_bps=cost)
        new = initial["paths"][cost]
        errors = {k: float(np.max(np.abs(new[k]-old[k]))) for k in ["target_weights", "pretrade_weights", "ending_weights", "fees", "net_equity"]}
        relative_nav = float(np.max(np.abs(new["equity"]/old["equity"]-1)))
        oldA = spec.risk_group_length/len(x)*float(np.sum(np.log(old["net_equity"])-np.log(old_anchor["net_equity"])))
        errors["F_component"] = abs(initial["A_by_cost"][cost]-oldA); Fs.append(oldA)
        if max(errors.values()) > 1e-12 or relative_nav > 1e-10:
            raise NumericalFitFailure("fresh initial parent-owned path differs")
        audit["costs"][cost] = {"maximum_absolute_errors": errors, "maximum_relative_nav_error": relative_nav}
    audit["worst_F_error"] = abs(initial["F"]-min(Fs))
    return audit


def _record(epoch, model, current, selected_epoch, used, next_dual, gradient=None, step_norm=0., pre_lagrangian=None):
    return {"epoch": epoch, "F": current["F"], "A_by_cost": current["A_by_cost"],
        "centered_worst_utility": current["F"], "centered_utility_by_cost": current["A_by_cost"],
        "daily_mean_log_excess_by_cost": current["daily_mean_log_excess_by_cost"],
        "aggregate_log_excess_by_cost": current["aggregate_log_excess_by_cost"],
        "legacy_J": current["legacy_J"], "legacy_centered_utility_by_cost": current["legacy_centered_utility_by_cost"],
        "cost_weights": current["cost_weights"], "feasible": current["feasible"],
        "risks_by_cost": {c: _risk_record(v) for c, v in current["constraints_by_cost"].items()},
        "lambda_used": used.tolist(), "postg": current["normalized_g"].tolist(), "lambda_next": next_dual.tolist(),
        "dual_update_applied": epoch > 0, "pre_primal_lagrangian": pre_lagrangian,
        "post_primal_lagrangian_at_lambda_used": float(-current["F"]+used@current["normalized_g"]),
        "primal_step_norm": float(step_norm), "selected_epoch_so_far": selected_epoch,
        "gradient": gradient, "action_diagnostics": {c: _path_diagnostics(p) for c, p in current["paths"].items()},
        "checkpoint": policy_to_dict(model)}


def fit_policy(features, returns, *, seed, spec: ConstrainedNetExcessSpec):
    x, r = _matrix(features, 27, "features"), _matrix(returns, 2, "returns")
    if len(x) != len(r) or len(x) < spec.minimum_rows or (r <= -1).any(): raise ValueError("blocked: invalid mature training inputs")
    history, failure, epoch, boundaries, optimizer, steps = [], None, "reference", None, None, 0
    initial_audit = None; references = {}; dual = np.zeros(4)
    with _cpu_threads(spec.torch_threads):
        learner, anchor = SparseHoldingsPolicy(seed=seed), zero_anchor(seed)
        initial_payload = policy_to_dict(learner)
        anchors = {c: numpy_rollout(anchor, x, r, cost_bps=c) for c in spec.cost_bps}
        anchor_audit = _audit_anchor_paths(anchors, r)
        incumbent, incumbent_eval, selected_epoch = anchor, _fallback_evaluation(anchors, spec.cost_bps), "anchor"
        try:
            references = {c: reference_risk(np.log(anchors[c]["net_equity"])) for c in spec.cost_bps}
            incumbent_eval = _evaluate(anchor, x, r, spec=spec, anchors=anchors, references=references)
            if incumbent_eval["F"] != 0 or not incumbent_eval["feasible"] or np.any(incumbent_eval["normalized_g"] != 0):
                raise NumericalFitFailure("exact sparse SOXX anchor failed")
            epoch = 0; current = _evaluate(learner, x, r, spec=spec, anchors=anchors, references=references)
            initial_audit = _initial_parent_parity(learner, x, r, spec=spec, initial=current)
            if feasible_improvement(current, incumbent_eval["F"]): incumbent, incumbent_eval, selected_epoch = deepcopy(learner), current, 0
            history.append(_record(0, learner, current, selected_epoch, dual, dual))
            optimizer = torch.optim.Adam(learner.parameters(), lr=spec.learning_rate, weight_decay=0.)
            for epoch in range(1, spec.updates+1):
                used = dual.copy(); pre_lagrangian = float(-current["F"]+used@current["normalized_g"])
                proxy, adjoint_audit = gradient_proxy(current, used, spec=spec)
                gradient = truncated_utility_gradient(learner, x, r, proxy, spec=spec)
                if boundaries is None: boundaries = gradient["chunk_boundaries"]
                elif boundaries != gradient["chunk_boundaries"]: raise NumericalFitFailure("gradient chunk coverage changed")
                gradient.pop("chunk_boundaries"); gradient.update(adjoint_audit); gradient["parameter_epoch"] = epoch-1
                gradient["unclipped_global_norm"] = float(torch.nn.utils.clip_grad_norm_(learner.parameters(), spec.gradient_clip))
                if not np.isfinite(gradient["unclipped_global_norm"]): raise NumericalFitFailure("nonfinite primal gradient norm")
                before = [p.detach().clone() for p in learner.parameters()]
                optimizer.step(); steps += 1  # No parameter projection: only forward sparsemax and dual projection.
                _validate_model(learner)
                step_norm = float(torch.sqrt(sum((p.detach()-b).square().sum() for p,b in zip(learner.parameters(), before))))
                current = _evaluate(learner, x, r, spec=spec, anchors=anchors, references=references)
                dual = update_multipliers(used, current["normalized_g"], learning_rate=spec.dual_learning_rate)
                if not np.isfinite([pre_lagrangian, step_norm, -current["F"]+used@current["normalized_g"]]).all():
                    raise NumericalFitFailure("nonfinite optimizer diagnostic")
                if feasible_improvement(current, incumbent_eval["F"]): incumbent, incumbent_eval, selected_epoch = deepcopy(learner), current, epoch
                history.append(_record(epoch, learner, current, selected_epoch, used, dual, gradient, step_norm, pre_lagrangian))
        except (NumericalFitFailure, TailConstraintError) as exc:
            failure = {"epoch": epoch, "reason": str(exc), "qualification_eligible": False, "accounting_anchor_only": True}
            incumbent, incumbent_eval, selected_epoch = anchor, _fallback_evaluation(anchors, spec.cost_bps), "anchor"
        incumbent.eval()
    groups = risk_groups(len(x), spec.risk_group_length)
    metadata = {"spec": asdict(spec), "seed": int(seed), "training_rows": len(x),
        "architecture": "27to8 sigmoid plus unconstrained4expert head and direct4x3 actual-holdings skip; sparsemax", "parameter_count": 272,
        "objective_name": "worst_cost_mean_net_log_excess_scaled63", "selected_epoch": selected_epoch,
        "failure": failure, "qualification_eligible": failure is None, "null_selected": is_exact_anchor(incumbent),
        "training_feasible": incumbent_eval["feasible"], "training_feasibility_passed": incumbent_eval["feasible"],
        "F": incumbent_eval["F"], "selected_F": incumbent_eval["F"], "centered_worst_utility": incumbent_eval["F"],
        "centered_utility_by_cost": incumbent_eval["A_by_cost"], "daily_mean_log_excess_by_cost": incumbent_eval["daily_mean_log_excess_by_cost"],
        "aggregate_log_excess_by_cost": incumbent_eval["aggregate_log_excess_by_cost"],
        "legacy_J": incumbent_eval["legacy_J"], "legacy_centered_utility_by_cost": incumbent_eval["legacy_centered_utility_by_cost"],
        "selected_risks_by_cost": {c: _risk_record(v) for c,v in incumbent_eval["constraints_by_cost"].items()},
        "selected_normalized_g": incumbent_eval["normalized_g"], "selected_action_diagnostics": {c: _path_diagnostics(p) for c,p in incumbent_eval["paths"].items()},
        "epochs": history, "history": history, "training_paths": incumbent_eval["paths"],
        "benchmark_net_equity": {c: anchors[c]["net_equity"] for c in spec.cost_bps},
        "benchmark_risk_normalizers": {c: v["normalizers"].tolist() for c,v in references.items()},
        "anchor_audit": anchor_audit, "initial_path_parity": initial_audit,
        "initial_tensor_sha256": initial_payload["tensor_sha256"], "parent_initial_tensor_sha256": learner.parent_initial_tensor_sha256,
        "training_feature_sha256": _hash_array(x), "training_return_sha256": _hash_array(r),
        "risk_groups": groups, "risk_group_overlap_observations": sum(b-a for a,b in groups)-len(x),
        "gradient_chunk_boundaries": boundaries, "final_optimizer_state": deepcopy(optimizer.state_dict()) if optimizer else None,
        "optimizer_initial_settings": deepcopy(optimizer.defaults) if optimizer else None,
        "optimizer_state_parameter_epoch": steps if optimizer else None, "optimizer_library_version": torch.__version__,
        "dual_order": ["10_drawdown", "10_cvar5", "25_drawdown", "25_cvar5"],
        "optimizer_diagnostics": {"updates_completed": steps, "finite_post_update_audits_completed": max(0,len(history)-1),
            "weight_decay": 0., "parameter_projection": False, "lambda_initial": [0.,0.,0.,0.], "lambda_final": dual.tolist(),
            "maximum_forward_parity_error": max((h["gradient"]["forward_parity_max_error"] for h in history if h["gradient"]),default=0.),
            "zero_gradient_update_count": sum(h["gradient"]["all_parameter_gradients_exact_zero"] for h in history if h["gradient"]),
            "incumbent_separate_from_optimizer_trajectory": True, "per_epoch_daily_adjoints_saved": False,
            "lagrangian_used_for_incumbent_ranking": False, "legacy_J_used_for_acceptance_or_gradient": False},
        "checkpoint_selection": "First strictly higher F with all four original raw tail gates; exact SOXX wins ties",
        "gradient_limits": "Strict-positive sparsemax support; implicit trade-sign0 funding generalized gradient;21context truncation; no global/composed Clarke or forward-risk guarantee",
        "no_reinforcement_learning": True}
    return incumbent, metadata


def evaluate_ensemble(models, features, returns, *, spec: ConstrainedNetExcessSpec):
    policies = list(models)
    if len(policies) != 3: raise ValueError("blocked: all three declared seed models required")
    x, r = _matrix(features, 27, "features"), _matrix(returns, 2, "returns")
    if len(x) != len(r) or not len(x) or (r <= -1).any(): raise ValueError("blocked: invalid ensemble panel")
    anchor = zero_anchor(policies[0].seed)
    anchors = {c: numpy_rollout(anchor, x, r, cost_bps=c) for c in spec.cost_bps}
    anchor_audit = _audit_anchor_paths(anchors, r); failure = None
    try:
        references = {c: reference_risk(np.log(anchors[c]["net_equity"])) for c in spec.cost_bps}
        result = _evaluate(policies, x, r, spec=spec, anchors=anchors, references=references)
        rejected = result["F"] < 0 or not result["feasible"]
        selected = [anchor] if rejected else policies
        chosen = _evaluate(selected, x, r, spec=spec, anchors=anchors, references=references) if rejected else result
    except (NumericalFitFailure, TailConstraintError) as exc:
        failure = {"reason": str(exc), "qualification_eligible": False, "accounting_anchor_only": True}
        rejected, selected = True, [anchor]; result = chosen = _fallback_evaluation(anchors, spec.cost_bps)
    return selected, {"component_count": 3, "all_three_seed_models_used": True, "failure": failure,
        "qualification_eligible": failure is None, "training_feasible": chosen["feasible"], "training_feasibility_passed": chosen["feasible"],
        "ensemble_F": result["F"], "ensemble_centered_utility": result["A_by_cost"], "ensemble_feasible": result["feasible"],
        "ensemble_legacy_J": result["legacy_J"], "ensemble_risks_by_cost": {c: _risk_record(v) for c,v in result["constraints_by_cost"].items()},
        "ensemble_rejected_for_training_utility_or_risk": rejected, "anchor_selected": all(is_exact_anchor(p) for p in selected),
        "null_selected": all(is_exact_anchor(p) for p in selected), "F": chosen["F"], "selected_F": chosen["F"],
        "centered_worst_utility": chosen["F"], "centered_utility_by_cost": chosen["A_by_cost"],
        "legacy_J": chosen["legacy_J"], "legacy_centered_utility_by_cost": chosen["legacy_centered_utility_by_cost"],
        "selected_risks_by_cost": {c: _risk_record(v) for c,v in chosen["constraints_by_cost"].items()},
        "selected_normalized_g": chosen["normalized_g"], "training_paths": chosen["paths"],
        "rejected_ensemble_training_paths": result["paths"] if rejected and failure is None else None,
        "selected_action_diagnostics": {c: _path_diagnostics(p) for c,p in chosen["paths"].items()},
        "anchor_audit": anchor_audit, "legacy_J_is_report_only": True}


__all__ = ["SparseHoldingsPolicy", "ConstrainedNetExcessSpec", "sparsemax", "sparsemax_numpy", "embed_parent_policy",
           "zero_anchor", "is_exact_anchor", "policy_to_dict", "policy_from_dict", "policy_target", "numpy_rollout", "replay_policy",
           "evaluate_policy", "evaluate_ensemble", "truncated_utility_gradient", "fit_policy"]
