"""Exact benchmark departures trained on continuous fee-funded utility.

Qlib Dataset maturity, feature scaling and source validity belong to the caller.
This module receives already standardized/masked chronological observations.
Forward accounting is exact to floating point; truncated state gradients and
nonsmooth fee subgradients are optimizer proposals, not convergence guarantees.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.model.integrated_allocation import _settle, _cpu_threads


class NumericalFitFailure(ValueError):
    """A declared numerical failure, never an economically selected null."""


@dataclass(frozen=True)
class AnchorUtilitySpec:
    updates: int = 120
    learning_rate: float = .003
    gradient_clip: float = 5.
    risk_group_length: int = 63
    gradient_chunk_length: int = 63
    context_sessions: int = 21
    cost_bps: tuple[float, ...] = (10., 25.)
    minimum_rows: int = 252
    torch_threads: int = 1
    downside_coefficient: float = 1.
    rms_epsilon: float = 1e-12

    def __post_init__(self):
        if (self.updates < 0 or self.risk_group_length < 1 or self.gradient_chunk_length < 1
                or self.context_sessions < 0 or self.minimum_rows < self.risk_group_length
                or self.torch_threads < 1):
            raise ValueError("blocked: invalid anchor training schedule")
        values = [self.learning_rate, self.gradient_clip, self.downside_coefficient, self.rms_epsilon]
        if not np.isfinite(values).all() or min(self.learning_rate, self.gradient_clip, self.rms_epsilon) <= 0 or self.downside_coefficient < 0:
            raise ValueError("blocked: invalid anchor utility parameters")
        if len(self.cost_bps) != 2 or len(set(self.cost_bps)) != 2:
            raise ValueError("blocked: two distinct declared cost scenarios required")
        for cost in self.cost_bps:
            _rate(cost)


def _rate(cost):
    if not np.isfinite(cost) or not 0 <= cost < 10000:
        raise ValueError("blocked: invalid proportional cost")
    return float(cost) / 10000


def _matrix(value, columns, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != columns or not np.isfinite(result).all():
        raise ValueError(f"blocked: {name} requires finite N x {columns} values")
    return np.ascontiguousarray(result)


def _weights(value, name="holdings"):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all() or result.min() < -1e-12 or abs(result.sum() - 1) > 1e-10:
        raise ValueError(f"blocked: invalid {name}")
    return result


def _validate_parameters(model):
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise NumericalFitFailure("nonfinite model parameters")
    theta = model.theta.detach()
    if bool((theta < 0).any()) or float(theta.sum()) > 1 + 1e-12:
        raise NumericalFitFailure("infeasible departure coefficients")


class BenchmarkAnchorPolicy(nn.Module):
    """Three expert departures with a globally bounded nonnegative budget."""
    feature_count = 27
    hidden_units = 8

    def __init__(self, *, seed: int = 17, anchor: bool = False):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.hidden = nn.Linear(self.feature_count, self.hidden_units, dtype=torch.float64)
        with torch.no_grad():
            self.hidden.bias.zero_()
        self.theta = nn.Parameter(torch.full((3, 9), 0. if anchor else 1 / 54, dtype=torch.float64))
        self.seed = int(seed)

    def alphas(self, features: torch.Tensor) -> torch.Tensor:
        z = self.hidden(features)
        if not bool(torch.isfinite(z).all()):
            raise NumericalFitFailure("nonfinite hidden preactivations")
        basis = torch.cat((torch.ones_like(z[..., :1]), torch.sigmoid(z)), dim=-1)
        return basis @ self.theta.T

    def forward(self, features: torch.Tensor, holdings: torch.Tensor) -> torch.Tensor:
        return _torch_target(self.alphas(features), holdings)

    def predict_target(self, features, pretrade_weights):
        return policy_target(self, features, pretrade_weights)


def _torch_target(alpha, held):
    left, cash, hold = alpha.unbind(-1)
    return torch.stack((left + hold * held[..., 0],
                        1 - alpha.sum(-1) + hold * held[..., 1],
                        cash + hold * held[..., 2]), dim=-1)


def project_theta(value: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto the nonnegative unit L1 ball."""
    if not bool(torch.isfinite(value).all()):
        raise NumericalFitFailure("nonfinite coefficient proposal")
    flat = value.flatten()
    projected = flat.clamp_min(0)
    if float(projected.sum()) > 1:
        ordered = torch.sort(flat, descending=True).values
        cumulative = ordered.cumsum(0) - 1
        divisor = torch.arange(1, flat.numel() + 1, dtype=flat.dtype, device=flat.device)
        eligible = ordered - cumulative / divisor > 0
        rho = int(torch.nonzero(eligible)[-1])
        projected = (flat - cumulative[rho] / (rho + 1)).clamp_min(0)
        # Only repair a possible final floating-point summation overshoot.
        if float(projected.sum()) > 1:
            projected = projected / projected.sum()
    return projected.reshape_as(value)


def policy_to_dict(model) -> dict[str, Any]:
    _validate_parameters(model)
    state = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(str(state[name].shape).encode())
        digest.update(state[name].tobytes())
    return {"schema_version": 1, "seed": model.seed, "feature_count": 27, "hidden_units": 8,
            "dtype": "float64", "tensor_sha256": digest.hexdigest(), "state_dict": state}


def policy_from_dict(payload) -> BenchmarkAnchorPolicy:
    if payload.get("schema_version") != 1 or payload.get("feature_count") != 27 or payload.get("hidden_units") != 8 or payload.get("dtype") != "float64":
        raise ValueError("blocked: incompatible anchor policy schema")
    model = BenchmarkAnchorPolicy(seed=int(payload["seed"]), anchor=True)
    model.load_state_dict({name: torch.as_tensor(value, dtype=torch.float64) for name, value in payload["state_dict"].items()})
    _validate_parameters(model)
    if policy_to_dict(model)["tensor_sha256"] != payload.get("tensor_sha256"):
        raise ValueError("blocked: policy tensor hash mismatch")
    return model.eval()


def numpy_alphas(models, features) -> np.ndarray:
    """Equal-target ensembles equal mean alphas on the same owned holdings."""
    x = _matrix(features, 27, "standardized features")
    policies = list(models) if isinstance(models, (list, tuple)) else [models]
    if not policies:
        raise ValueError("blocked: empty anchor ensemble")
    result = np.zeros((len(x), 3))
    for model in policies:
        _validate_parameters(model)
        p = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
        z = x @ p["hidden.weight"].T + p["hidden.bias"]
        if not np.isfinite(z).all():
            raise NumericalFitFailure("nonfinite numpy preactivations")
        gate = np.empty_like(z)
        positive = z >= 0
        gate[positive] = 1 / (1 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        gate[~positive] = exp_z / (1 + exp_z)
        result += np.c_[np.ones(len(x)), gate] @ p["theta"].T / len(policies)
    if not np.isfinite(result).all() or result.min(initial=0) < 0 or (result.sum(axis=1) > 1 + 1e-12).any():
        raise NumericalFitFailure("invalid expert mixture")
    return result


def policy_target(models, features, pretrade_weights):
    h = _weights(pretrade_weights)
    a = numpy_alphas(models, np.asarray(features, dtype=float).reshape(1, -1))[0]
    return np.array([a[0] + a[2] * h[0], 1 - a.sum() + a[2] * h[1], a[1] + a[2] * h[2]])


def _scalar_settle(h0, h1, h2, z0, z1, z2, r0, r1, rate):
    """Four exact sign roots, scalar arithmetic to avoid tiny per-day arrays."""
    if (z0 == h0 and z1 == h1 and z2 == h2) or rate == 0:
        net_open = 1.
    else:
        best = math.inf
        net_open = math.nan
        for s0, s1 in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            candidate = (1 + rate * (s0 * h0 + s1 * h1)) / (1 + rate * (s0 * z0 + s1 * z1))
            error = abs(candidate + rate * (abs(candidate * z0 - h0) + abs(candidate * z1 - h1)) - 1)
            if error < best:
                best, net_open = error, candidate
    x0, x1 = net_open * z0, net_open * z1
    turnover = abs(x0 - h0) + abs(x1 - h1)
    fee = rate * turnover
    executable_cash = 1 - (x0 + x1)
    x2 = executable_cash - fee
    v0, v1 = x0 * (1 + r0), x1 * (1 + r1)
    net = v0 + v1 + x2
    if not math.isfinite(net) or net <= 0 or x2 < -1e-12:
        raise NumericalFitFailure("nonpositive NAV or invalid cash funding")
    return (v0 / net, v1 / net, x2 / net), net, fee, turnover, (x0, x1, executable_cash), (x0, x1, x2)


def rollout_alphas(alphas, returns, *, cost_bps: float, initial_weights=None) -> dict[str, np.ndarray]:
    """One chronological fee-funded path; does not infer validity from future rows."""
    a = _matrix(alphas, 3, "expert coefficients")
    r = _matrix(returns, 2, "asset returns")
    if len(a) != len(r) or len(a) == 0 or (a < 0).any() or (a.sum(1) > 1 + 1e-12).any() or (r <= -1).any():
        raise ValueError("blocked: invalid chronological alpha/return panel")
    rate = _rate(cost_bps)
    held = _weights([0., 0., 1.] if initial_weights is None else initial_weights)
    n = len(a)
    result = {k: np.empty((n, 3)) for k in ["pretrade_weights", "target_weights", "ending_weights", "open_net_weights", "execution_target_weights"]}
    result.update({k: np.empty(n) for k in ["net_equity", "fees", "turnover", "open_net_equity"]})
    h0, h1, h2 = map(float, held)
    for i in range(n):
        left, cash, hold = map(float, a[i])
        z0, z1, z2 = left + hold * h0, 1 - (left + cash + hold) + hold * h1, cash + hold * h2
        result["pretrade_weights"][i] = h0, h1, h2
        result["target_weights"][i] = z0, z1, z2
        ending, net, fee, turnover, execution, opened = _scalar_settle(h0, h1, h2, z0, z1, z2, float(r[i, 0]), float(r[i, 1]), rate)
        open_net = sum(opened)
        result["ending_weights"][i] = ending
        result["open_net_weights"][i] = tuple(x / open_net for x in opened)
        result["execution_target_weights"][i] = execution
        result["net_equity"][i], result["fees"][i], result["turnover"][i], result["open_net_equity"][i] = net, fee, turnover, open_net
        h0, h1, h2 = ending
    result["net_return"] = result["net_equity"] - 1
    result["equity"] = result["net_equity"].cumprod()
    if not all(np.isfinite(value).all() for value in result.values()) or (result["equity"] <= 0).any():
        raise NumericalFitFailure("nonfinite or nonpositive chronological wealth")
    return result


def numpy_rollout(models, features, returns, *, cost_bps, initial_weights=None, force_anchor_mask=None):
    alpha = numpy_alphas(models, features)
    if force_anchor_mask is not None:
        mask = np.asarray(force_anchor_mask)
        if mask.dtype != np.bool_ or mask.shape != (len(alpha),):
            raise ValueError("blocked: anchor mask must be explicit per-date boolean")
        alpha[mask] = 0.
    return rollout_alphas(alpha, returns, cost_bps=cost_bps, initial_weights=initial_weights)


def replay_policy(models, features, returns, *, cost_bps, initial_weights=None, force_anchor_mask=None):
    arrays = numpy_rollout(models, features, returns, cost_bps=cost_bps, initial_weights=initial_weights, force_anchor_mask=force_anchor_mask)
    frame = pd.DataFrame({"strategy_return": arrays["net_return"], "equity": arrays["equity"], "turnover": arrays["turnover"],
                          "fees": arrays["fees"], "cost_bps": float(cost_bps), "execute_open_target": True,
                          "execute_open_trade": arrays["turnover"] > 1e-12})
    for prefix, name in [("", "open_net_weights"), ("pretrade_", "pretrade_weights"), ("ending_", "ending_weights"), ("execution_", "execution_target_weights")]:
        for j, asset in enumerate(["soxl", "soxx", "cash"]):
            frame[prefix + asset + "_weight"] = arrays[name][:, j]
    return frame


def risk_groups(length: int, group_length: int = 63) -> list[tuple[int, int]]:
    if length < group_length or group_length < 1:
        raise ValueError("blocked: insufficient complete risk group")
    starts = list(range(0, length - group_length + 1, group_length))
    if starts[-1] != length - group_length:
        starts.append(length - group_length)
    return [(start, start + group_length) for start in starts]


def utility_and_daily_weights(log_relative, *, spec: AnchorUtilitySpec) -> dict[str, Any]:
    x = np.asarray(log_relative, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise NumericalFitFailure("invalid complete log-relative path")
    groups = risk_groups(len(x), spec.risk_group_length)
    sums = np.array([x[start:end].sum() for start, end in groups])
    downside = np.minimum(sums, 0.)
    rms = float(np.sqrt(np.mean(downside ** 2) + spec.rms_epsilon))
    weights = np.full(len(x), spec.risk_group_length / len(x))
    for (start, end), negative in zip(groups, downside):
        weights[start:end] -= spec.downside_coefficient * negative / (len(groups) * rms)
    value = spec.risk_group_length / len(x) * x.sum() - spec.downside_coefficient * rms
    if not np.isfinite(value) or not np.isfinite(weights).all():
        raise NumericalFitFailure("nonfinite global utility or derivative")
    return {"utility": float(value), "daily_weights": weights, "risk_groups": groups,
            "group_log_excess": sums, "downside_rms": rms, "total_log_excess": float(x.sum()),
            "global_mean_observations": len(x), "risk_memberships": sum(end - start for start, end in groups)}


def _evaluate_alphas(alpha, returns, spec, anchors):
    paths, components = {}, {}
    for cost in spec.cost_bps:
        path = rollout_alphas(alpha, returns, cost_bps=cost)
        relative = np.log(path["net_equity"]) - np.log(anchors[cost]["net_equity"])
        paths[cost] = path
        components[cost] = utility_and_daily_weights(relative, spec=spec)
    anchor_utility = -spec.downside_coefficient * math.sqrt(spec.rms_epsilon)
    centered = {cost: components[cost]["utility"] - anchor_utility for cost in spec.cost_bps}
    worst = min(centered.values())
    active = [cost for cost, value in centered.items() if value == worst]
    weights = {cost: 1 / len(active) if cost in active else 0. for cost in spec.cost_bps}
    return {"J": float(worst), "centered_utility": centered, "cost_weights": weights,
            "components": components, "paths": paths, "anchor_utility": anchor_utility}


def _audit_anchor_paths(anchors, returns):
    """Independent analytic fully funded SOXX entry followed by exact HOLD."""
    audits = {}
    for cost, path in anchors.items():
        rate = _rate(cost)
        reference = 1 + returns[:, 1].copy()
        reference[0] /= 1 + rate
        reference_wealth = reference.cumprod()
        multiplier_error = float(np.max(np.abs(reference - path["net_equity"])))
        relative_wealth_error = float(np.max(np.abs(path["equity"] / reference_wealth - 1)))
        fee_error = abs(float(path["fees"][0]) - rate / (1 + rate))
        later_fees = float(np.max(np.abs(path["fees"][1:]), initial=0))
        if (not np.isfinite([multiplier_error, relative_wealth_error, fee_error, later_fees]).all()
                or max(multiplier_error, relative_wealth_error, fee_error, later_fees) > 1e-10):
            raise NumericalFitFailure("anchor differs from independent funded SOXX buy-and-hold")
        audits[cost] = {"maximum_multiplier_error": multiplier_error,
                        "maximum_relative_wealth_error": relative_wealth_error,
                        "initial_entry_fee_error": fee_error, "maximum_later_fee": later_fees,
                        "tolerance": 1e-10, "initial_weights": [0., 0., 1.],
                        "reference": "first g=(1+r_SOXX)/(1+c); later g=1+r_SOXX; no later orders"}
    return audits


def evaluate_policy(models, features, returns, *, spec: AnchorUtilitySpec, anchors=None):
    x, r = _matrix(features, 27, "features"), _matrix(returns, 2, "returns")
    if len(x) != len(r):
        raise ValueError("blocked: features and returns differ in length")
    anchors = anchors or {cost: rollout_alphas(np.zeros((len(r), 3)), r, cost_bps=cost) for cost in spec.cost_bps}
    return _evaluate_alphas(numpy_alphas(models, x), r, spec, anchors)


def truncated_utility_gradient(model, features, returns, evaluation, *, spec: AnchorUtilitySpec):
    """Accumulate one global-utility gradient; do not step inside chunk loops.

    Independent NumPy full paths supply detached current-policy boundary states.
    Gates are recomputed WITH autograd, so W/b receive their true local route.
    """
    x, r = _matrix(features, 27, "features"), _matrix(returns, 2, "returns")
    if len(x) != len(r) or (r <= -1).any():
        raise ValueError("blocked: invalid gradient return panel")
    n = len(x)
    starts = list(range(0, n, spec.gradient_chunk_length))
    boundaries = [(max(0, start - spec.context_sessions), start, min(n, start + spec.gradient_chunk_length)) for start in starts]
    longest = max(end - context for context, start, end in boundaries)
    indices = np.zeros((len(boundaries), longest), dtype=np.int64)
    valid = np.zeros_like(indices, dtype=bool)
    scored = np.zeros_like(indices, dtype=bool)
    for row, (context, start, end) in enumerate(boundaries):
        size = end - context
        indices[row, :size] = np.arange(context, end)
        valid[row, :size] = True
        scored[row, start - context:size] = True
    assert int(scored.sum()) == n
    model.zero_grad(set_to_none=True)
    alpha = model.alphas(torch.from_numpy(x))  # Never reuse detached forward gates.
    alpha_chunks = alpha[torch.from_numpy(indices)]
    ret = torch.cat((torch.from_numpy(r), torch.zeros((n, 1), dtype=torch.float64)), dim=-1)[indices]
    active_mask = torch.from_numpy(valid)
    loss = alpha.sum() * 0.
    max_parity = 0.
    for cost in spec.cost_bps:
        full = evaluation["paths"][cost]
        held = torch.from_numpy(np.array([full["pretrade_weights"][context] for context, _, _ in boundaries]))
        adjoints = evaluation["components"][cost]["daily_weights"] * evaluation["cost_weights"][cost]
        weights = torch.from_numpy(adjoints[indices] * scored)
        for t in range(longest):
            active = active_mask[:, t]
            target = _torch_target(alpha_chunks[:, t], held)
            target = torch.where(active[:, None], target, held)
            returns_t = torch.where(active[:, None], ret[:, t], torch.zeros_like(ret[:, t]))
            settled = _settle(held, target, returns_t, _rate(cost))
            net = torch.where(active, settled["net_equity"], torch.ones_like(settled["net_equity"]))
            if not bool(torch.isfinite(net).all()) or bool((net <= 0).any()):
                raise NumericalFitFailure("nonfinite or nonpositive differentiable NAV")
            observed = net.detach().numpy()[valid[:, t]]
            expected = full["net_equity"][indices[:, t][valid[:, t]]]
            max_parity = max(max_parity, float(np.max(np.abs(observed - expected), initial=0)))
            loss = loss - (weights[:, t] * torch.log(net)).sum()
            held = torch.where(active[:, None], settled["ending_weights"], held)
    if max_parity > 1e-10:
        raise NumericalFitFailure(f"scalar/Torch current-path mismatch: {max_parity}")
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise NumericalFitFailure("missing or nonfinite optimizer gradient")
    return {"surrogate_loss": float(loss.detach()), "forward_parity_max_error": max_parity,
            "scored_rows": int(scored.sum()), "chunk_boundaries": boundaries,
            "theta_gradient_norm": float(model.theta.grad.norm()),
            "hidden_gradient_norm": float(torch.sqrt(sum(p.grad.square().sum() for p in model.hidden.parameters())))}


def _epoch_record(epoch, model, evaluation, selected_epoch, gradient=None):
    theta = model.theta.detach().numpy()
    return {"epoch": epoch, "J": evaluation["J"], "centered_utility": evaluation["centered_utility"],
            "cost_weights": evaluation["cost_weights"], "theta_mass": float(theta.sum()),
            "theta_active_coordinates": int((theta > 0).sum()), "theta_minimum": float(theta.min()),
            "selected_epoch_so_far": selected_epoch, "gradient": gradient,
            "checkpoint": policy_to_dict(model)}


def fit_policy(features, returns, *, seed: int, spec: AnchorUtilitySpec):
    """Fresh one-start projected-Adam fit with an exact full-history incumbent."""
    x, r = _matrix(features, 27, "features"), _matrix(returns, 2, "returns")
    if len(x) != len(r) or len(x) < spec.minimum_rows or (r <= -1).any():
        raise ValueError("blocked: insufficient or invalid mature chronological rows")
    with _cpu_threads(spec.torch_threads):
        learner = BenchmarkAnchorPolicy(seed=seed)
        anchor = BenchmarkAnchorPolicy(seed=seed, anchor=True)
        anchors = {cost: rollout_alphas(np.zeros((len(r), 3)), r, cost_bps=cost) for cost in spec.cost_bps}
        anchor_audit = _audit_anchor_paths(anchors, r)
        incumbent, incumbent_eval, selected_epoch = anchor, evaluate_policy(anchor, x, r, spec=spec, anchors=anchors), -1
        if incumbent_eval["J"] != 0 or any(value != 0 for value in incumbent_eval["centered_utility"].values()):
            raise NumericalFitFailure("exact anchor does not have zero centered utility")
        history, failure, epoch = [], None, 0
        try:
            current = evaluate_policy(learner, x, r, spec=spec, anchors=anchors)
            if current["J"] > incumbent_eval["J"]:
                incumbent, incumbent_eval, selected_epoch = deepcopy(learner), current, 0
            history.append(_epoch_record(0, learner, current, selected_epoch))
            optimizer = torch.optim.Adam(learner.parameters(), lr=spec.learning_rate, weight_decay=0.)
            for epoch in range(1, spec.updates + 1):
                gradient = truncated_utility_gradient(learner, x, r, current, spec=spec)
                gradient["parameter_epoch"] = epoch - 1
                gradient["unclipped_global_norm"] = float(torch.nn.utils.clip_grad_norm_(learner.parameters(), spec.gradient_clip))
                if not np.isfinite(gradient["unclipped_global_norm"]):
                    raise NumericalFitFailure("nonfinite global optimizer gradient norm")
                optimizer.step()
                with torch.no_grad():
                    learner.theta.copy_(project_theta(learner.theta))
                _validate_parameters(learner)
                current = evaluate_policy(learner, x, r, spec=spec, anchors=anchors)
                if current["J"] > incumbent_eval["J"]:
                    incumbent, incumbent_eval, selected_epoch = deepcopy(learner), current, epoch
                history.append(_epoch_record(epoch, learner, current, selected_epoch, gradient))
        except NumericalFitFailure as exc:
            failure = {"epoch": epoch, "reason": str(exc), "accounting_anchor_only": True, "qualification_eligible": False}
            incumbent, incumbent_eval, selected_epoch = anchor, evaluate_policy(anchor, x, r, spec=spec, anchors=anchors), -1
        incumbent.eval()
    metadata = {"spec": asdict(spec), "seed": int(seed), "training_rows": len(x), "selected_epoch": "anchor" if selected_epoch == -1 else selected_epoch,
                "null_selected": bool(torch.count_nonzero(incumbent.theta) == 0), "failure": failure,
                "qualification_eligible": failure is None, "history": history, "anchor_J": 0.,
                "anchor_audit": anchor_audit,
                "selected_J": incumbent_eval["J"], "selected_centered_utility": incumbent_eval["centered_utility"],
                "centered_worst_utility": incumbent_eval["J"], "centered_utility_by_cost": incumbent_eval["centered_utility"],
                "epochs": history,
                "training_paths": incumbent_eval["paths"], "benchmark_net_equity": {c: anchors[c]["net_equity"] for c in spec.cost_bps},
                "risk_groups": risk_groups(len(x), spec.risk_group_length),
                "risk_group_overlap_observations": sum(b-a for a,b in risk_groups(len(x), spec.risk_group_length)) - len(x),
                "training_feature_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                "training_return_sha256": hashlib.sha256(r.tobytes()).hexdigest(),
                "feature_scaling": "caller-provided prior-only standardized and masked27; no internal scaler",
                "gradient_scope": "global daily utility adjoints, detached true boundary holdings, up to21 prior context; truncated state derivatives",
                "fee_derivative_limit": "exact forward sign-root selection; autograd at no-trade is not certified as the feasible one-sided derivative",
                "optimizer": "projected Adam, no weight decay; no global convergence claim",
                "checkpoint_selection": "strict improvement of full-history worst-cost centered utility; anchor wins ties",
                "optimizer_diagnostics": {"updates_completed": max(0, len(history)-1),
                    "initial_theta_mass": .5, "fresh_initialization": True, "weight_decay": 0.,
                    "incumbent_separate_from_optimizer_trajectory": True,
                    "maximum_forward_parity_error": max((h["gradient"]["forward_parity_max_error"] for h in history if h["gradient"]), default=0.),
                    "nonsmooth_boundary_derivatives_certified": False,
                    "full_state_gradient_certified": False},
                "no_reinforcement_learning": True}
    return incumbent, metadata


def evaluate_ensemble(models, features, returns, *, spec: AnchorUtilitySpec):
    """Audit the equal-target ensemble on its own path; never select a subset."""
    policies = list(models)
    if len(policies) != 3:
        raise ValueError("blocked: exactly all three declared seed models required")
    result = evaluate_policy(policies, features, returns, spec=spec)
    rejected = result["J"] < 0
    selected = [BenchmarkAnchorPolicy(seed=policies[0].seed, anchor=True)] if rejected else policies
    selected_eval = evaluate_policy(selected, features, returns, spec=spec) if rejected else result
    return selected, {"ensemble_J": result["J"], "ensemble_centered_utility": result["centered_utility"],
                      "ensemble_rejected_for_training_utility": rejected, "selected_J": selected_eval["J"],
                      "selected_centered_utility": selected_eval["centered_utility"], "training_paths": selected_eval["paths"],
                      "centered_worst_utility": selected_eval["J"], "centered_utility_by_cost": selected_eval["centered_utility"],
                      "anchor_selected": all(bool(torch.count_nonzero(m.theta) == 0) for m in selected), "component_count": 3,
                      "rejected_ensemble_training_paths": result["paths"] if rejected else None,
                      "all_three_seed_models_used": True, "null_selected": all(bool(torch.count_nonzero(m.theta) == 0) for m in selected)}


__all__ = ["AnchorUtilitySpec", "BenchmarkAnchorPolicy", "NumericalFitFailure", "fit_policy", "evaluate_ensemble",
           "numpy_alphas", "rollout_alphas", "numpy_rollout", "replay_policy", "policy_target", "project_theta",
           "policy_to_dict", "policy_from_dict", "risk_groups", "utility_and_daily_weights", "evaluate_policy",
           "truncated_utility_gradient"]
