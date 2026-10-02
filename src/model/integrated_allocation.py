"""Small supervised allocation model with differentiable, cash-funded costs.

Inputs are prepared by the Qlib Dataset workflow. Training differentiates a
fixed, chronological return panel directly: there is no exploration, critic,
policy-gradient estimator or reinforcement-learning environment. The model
outputs unrestricted long-only SOXL/SOXX/cash simplex weights. Fees are paid
from cash before returns, on the actual sum of risky-asset purchases and sales.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import hashlib
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.workflow.direct_continuous_soxx_allocation import DirectContinuousAllocationNet


@dataclass(frozen=True)
class AllocationSpec:
    """All experiment choices must be frozen in the notebook's registration."""
    hidden_units: int = 8
    epochs: int = 120
    learning_rate: float = .003
    weight_decay: float = .01
    block_length: int = 63
    warmup_sessions: int = 21
    batch_blocks: int = 32
    downside_coefficient: float = 1.
    auxiliary_coefficient: float = 0.
    training_cost_bps: float = 10.
    gradient_clip: float = 5.
    torch_threads: int = 1

    def __post_init__(self):
        if self.hidden_units < 1 or self.epochs < 1 or self.block_length < 2 or self.warmup_sessions < 0 or self.batch_blocks < 1 or self.torch_threads < 1:
            raise ValueError("blocked: invalid allocation architecture or training length")
        scalars = (self.learning_rate, self.weight_decay, self.downside_coefficient,
                   self.auxiliary_coefficient, self.training_cost_bps, self.gradient_clip)
        if not np.isfinite(scalars).all() or self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("blocked: invalid allocation training parameters")
        if min(self.weight_decay, self.downside_coefficient, self.auxiliary_coefficient, self.training_cost_bps) < 0 or self.training_cost_bps >= 10000:
            raise ValueError("blocked: invalid allocation costs or objective coefficients")


def _validate_cost(cost_bps: float) -> float:
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("blocked: invalid proportional allocation cost")
    return float(cost_bps) / 10000.


def _validate_weights(weights: torch.Tensor, name: str) -> None:
    if weights.shape[-1:] != (3,) or not weights.is_floating_point():
        raise ValueError(f"blocked: {name} must have three floating-point asset weights")
    tolerance = 1e-10 if weights.dtype == torch.float64 else 2e-6
    if not bool(torch.isfinite(weights).all()) or bool((weights < -tolerance).any()):
        raise ValueError(f"blocked: nonfinite or short {name}")
    if not bool(torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), rtol=0, atol=tolerance)):
        raise ValueError(f"blocked: unnormalized {name}")


def _settle(pretrade: torch.Tensor, target: torch.Tensor | None,
            returns: torch.Tensor, rate: float) -> dict[str, torch.Tensor]:
    """Unchecked vectorized kernel; solve all four risky-trade sign regions."""
    if target is None:
        open_values = pretrade
        turnover = torch.zeros_like(pretrade[..., 0])
        fees = turnover
        executable = pretrade
    else:
        signs = pretrade.new_tensor(((-1., -1.), (-1., 1.), (1., -1.), (1., 1.)))
        current_risky, desired_risky = pretrade[..., None, :2], target[..., None, :2]
        # For each sign s: v=(1+c*s.w)/(1+c*s.z). Exactly one region
        # applies away from a no-trade boundary; all valid boundary roots agree.
        candidates = (1 + rate * (signs * current_risky).sum(-1)) / (1 + rate * (signs * desired_risky).sum(-1))
        residual = (candidates + rate * (candidates[..., None] * desired_risky - current_risky).abs().sum(-1) - 1).abs()
        selected = residual.detach().argmin(-1, keepdim=True)
        net_open = candidates.gather(-1, selected).squeeze(-1)
        risky_values = net_open[..., None] * target[..., :2]
        turnover = (risky_values - pretrade[..., :2]).abs().sum(-1)
        fees = rate * turnover
        executable = torch.cat((risky_values, (1 - risky_values.sum(-1))[..., None]), dim=-1)
        # Include the explicit cash debit, preserving the same identity as the
        # NumPy settlement implementation rather than treating fees as returns.
        open_values = torch.cat((risky_values, (executable[..., 2] - fees)[..., None]), dim=-1)
    ending_values = open_values * (1 + returns)
    net_equity = ending_values.sum(-1)
    open_net_equity = open_values.sum(-1)
    return {
        "net_equity": net_equity, "net_return": net_equity - 1,
        "gross_return": net_equity + fees - 1, "ending_values": ending_values,
        "ending_weights": ending_values / net_equity[..., None],
        "turnover": turnover, "open_turnover": turnover, "fees": fees,
        "open_net_equity": open_net_equity,
        "open_net_weights": open_values / open_net_equity[..., None],
        "execution_target_weights": executable,
    }


def differentiable_settlement(pretrade_weights: torch.Tensor, target_net_weights: torch.Tensor | None,
                              asset_returns: torch.Tensor, *, cost_bps: float) -> dict[str, torch.Tensor]:
    """Batch-compatible equivalent of fee_affordable_target + settle_target_session.

    ``None`` is an exact HOLD. A supplied target denotes desired post-fee net
    weights, not pre-fee order weights. Inputs share shape ``(..., 3)``.
    """
    rate = _validate_cost(cost_bps)
    _validate_weights(pretrade_weights, "pretrade weights")
    if asset_returns.shape != pretrade_weights.shape or asset_returns.dtype != pretrade_weights.dtype or asset_returns.device != pretrade_weights.device:
        raise ValueError("blocked: returns and holdings must share shape, dtype and device")
    if not bool(torch.isfinite(asset_returns).all()) or bool((asset_returns <= -1).any()):
        raise ValueError("blocked: invalid allocation asset returns")
    if target_net_weights is not None:
        _validate_weights(target_net_weights, "net target")
        if target_net_weights.shape != pretrade_weights.shape or target_net_weights.dtype != pretrade_weights.dtype or target_net_weights.device != pretrade_weights.device:
            raise ValueError("blocked: allocation targets and holdings differ in shape, dtype or device")
    return _settle(pretrade_weights, target_net_weights, asset_returns, rate)


class IntegratedAllocationPolicy(nn.Module):
    """The existing small Tanh network, plus actual holdings as three inputs."""
    def __init__(self, feature_count: int, hidden_units: int = 16, *,
                 common_feature_indices: tuple[int, ...] = (), auxiliary_target_count: int = 0):
        super().__init__()
        if feature_count < 1 or hidden_units < 1:
            raise ValueError("blocked: empty allocation feature space")
        indices = tuple(int(index) for index in common_feature_indices)
        if len(set(indices)) != len(indices) or any(index < 0 or index >= feature_count for index in indices):
            raise ValueError("blocked: invalid shared common-feature indices")
        if auxiliary_target_count < 0 or (auxiliary_target_count > 0 and not indices):
            raise ValueError("blocked: auxiliary targets require explicit common features")
        self.feature_count = int(feature_count)
        self.hidden_units = int(hidden_units)
        self.common_feature_indices = indices
        self.allocator = DirectContinuousAllocationNet(feature_count + 3, hidden_units)
        self.register_buffer("feature_mean", torch.zeros(feature_count))
        self.register_buffer("feature_scale", torch.ones(feature_count))
        self.auxiliary_head = nn.Linear(hidden_units, auxiliary_target_count) if auxiliary_target_count else None
        self.register_buffer("auxiliary_target_mean", torch.zeros(auxiliary_target_count))
        self.register_buffer("auxiliary_target_scale", torch.ones(auxiliary_target_count))

    def forward(self, features: torch.Tensor, pretrade_weights: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        return torch.softmax(self.allocator(torch.cat((normalized, pretrade_weights), dim=-1)), dim=-1)

    def auxiliary_prediction(self, common_features: torch.Tensor) -> torch.Tensor:
        """Shared common projection; no fabricated SOXL fields or holdings."""
        if self.auxiliary_head is None or common_features.shape[-1] != len(self.common_feature_indices):
            raise ValueError("blocked: auxiliary head or common features unavailable")
        indices = list(self.common_feature_indices)
        normalized = (common_features - self.feature_mean[indices]) / self.feature_scale[indices]
        first = self.allocator.network[0]
        hidden = torch.tanh(F.linear(normalized, first.weight[:, indices], first.bias))
        return self.auxiliary_head(hidden)

    def predict_target(self, features, pretrade_weights) -> np.ndarray:
        return policy_target(self, features, pretrade_weights)


def policy_target(models, features, pretrade_weights) -> np.ndarray:
    """Equal-weight model ensemble, all seeing the same actual holdings."""
    policies = list(models) if isinstance(models, (list, tuple)) else [models]
    if not policies:
        raise ValueError("blocked: empty allocation ensemble")
    targets = []
    with torch.no_grad():
        for model in policies:
            parameter = next(model.parameters())
            x = _tensor(features, like=parameter)
            weights = _tensor(pretrade_weights, like=parameter)
            if x.shape != (model.feature_count,) or weights.shape != (3,) or not bool(torch.isfinite(x).all()):
                raise ValueError("blocked: invalid allocation inference features or holdings")
            _validate_weights(weights, "inference holdings")
            targets.append(model(x, weights).detach().cpu().numpy().astype(np.float64))
    target = np.mean(targets, axis=0)
    if not np.isfinite(target).all() or (target < 0).any() or target.sum() <= 0:
        raise ValueError("blocked: allocation model produced an invalid simplex")
    return target / target.sum()


@contextmanager
def _cpu_threads(count: int):
    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _tensor(value, *, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def rollout_policy(model: IntegratedAllocationPolicy, features: torch.Tensor,
                   asset_returns: torch.Tensor, *, cost_bps: float,
                   initial_weights: torch.Tensor | None = None, mask: torch.Tensor | None = None,
                   detach_state: bool = False) -> dict[str, torch.Tensor]:
    """Causal sequential rollout; supports ``[T,F]`` or batch ``[B,T,F]``.

    The day-t target uses features and holdings before day-t return is applied.
    Missing/padded observations require an explicit false mask; they cannot
    trade or alter holdings. The mask does not permit nonfinite actual inputs.
    """
    parameter = next(model.parameters())
    features, asset_returns = _tensor(features, like=parameter), _tensor(asset_returns, like=parameter)
    squeeze = features.ndim == 2
    if squeeze:
        features, asset_returns = features.unsqueeze(0), asset_returns.unsqueeze(0)
    if features.ndim != 3 or features.shape[-1] != model.feature_count or asset_returns.shape[:2] != features.shape[:2] or asset_returns.shape[-1] not in {2, 3} or features.shape[1] == 0:
        raise ValueError("blocked: incompatible chronological allocation inputs")
    if asset_returns.shape[-1] == 2:
        asset_returns = torch.cat((asset_returns, torch.zeros_like(asset_returns[..., :1])), dim=-1)
    if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(asset_returns).all()) or bool((asset_returns <= -1).any()):
        raise ValueError("blocked: nonfinite or invalid chronological allocation data")
    rate = _validate_cost(cost_bps)
    batch, periods, _ = features.shape
    if initial_weights is None:
        weights = features.new_tensor([0., 0., 1.]).expand(batch, -1)
    else:
        weights = _tensor(initial_weights, like=parameter)
        if weights.ndim == 1:
            weights = weights.expand(batch, -1)
        if weights.shape != (batch, 3):
            raise ValueError("blocked: invalid rollout initial holdings")
    _validate_weights(weights, "initial holdings")
    valid = torch.ones((batch, periods), dtype=torch.bool, device=features.device) if mask is None else torch.as_tensor(mask, dtype=torch.bool, device=features.device)
    if squeeze and valid.ndim == 1:
        valid = valid.unsqueeze(0)
    if valid.shape != (batch, periods):
        raise ValueError("blocked: invalid allocation mask")
    if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError("blocked: masked training paths must be contiguous prefixes")
    fields = {name: [] for name in ("target_weights", "pretrade_weights", "ending_weights", "execution_target_weights",
              "net_return", "gross_return", "turnover", "fees", "net_equity", "open_net_equity")}
    for step in range(periods):
        active = valid[:, step]
        target = model(features[:, step], weights)
        target = torch.where(active[:, None], target, weights)
        returns = torch.where(active[:, None], asset_returns[:, step], torch.zeros_like(asset_returns[:, step]))
        settled = _settle(weights, target, returns, rate)
        fields["target_weights"].append(target)
        fields["pretrade_weights"].append(weights)
        for name in fields.keys() - {"target_weights", "pretrade_weights"}:
            fields[name].append(settled[name])
        weights = settled["ending_weights"].detach() if detach_state else settled["ending_weights"]
    result = {name: torch.stack(values, dim=1) for name, values in fields.items()}
    result["mask"] = valid
    result["equity"] = result["net_equity"].cumprod(dim=1)
    return {name: values.squeeze(0) for name, values in result.items()} if squeeze else result


def replay_policy(model: IntegratedAllocationPolicy, features, asset_returns, *, cost_bps: float = 10.,
                  initial_weights=None, startcash: float = 1.) -> pd.DataFrame:
    if not np.isfinite(startcash) or startcash <= 0:
        raise ValueError("blocked: invalid allocation replay capital")
    model.eval()
    with torch.no_grad():
        result = rollout_policy(model, features, asset_returns, cost_bps=cost_bps, initial_weights=initial_weights)
    if result["net_return"].ndim != 1:
        raise ValueError("blocked: public replay requires one continuous path")
    array = {name: value.detach().cpu().numpy() for name, value in result.items()}
    frame = pd.DataFrame({"strategy_return": array["net_return"], "gross_return": array["gross_return"],
        "turnover": array["turnover"], "open_turnover": array["turnover"], "fees": array["fees"],
        "cost_bps": float(cost_bps), "equity": startcash * array["equity"],
        "execute_open_trade": array["turnover"] > 1e-12})
    for index, asset in enumerate(("soxl", "soxx", "cash")):
        for prefix, key in (("", "target_weights"), ("pretrade_", "pretrade_weights"),
                            ("ending_", "ending_weights"), ("execution_", "execution_target_weights")):
            frame[f"{prefix}{asset}_weight"] = array[key][:, index]
    return frame


def _array(value, *, name: str, columns: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or (columns is not None and result.shape[1] != columns) or not np.isfinite(result).all():
        raise ValueError(f"blocked: {name} must be a finite two-dimensional matrix")
    return result


def _blocks(features: torch.Tensor, returns: torch.Tensor, benchmark: torch.Tensor,
            block_length: int, warmup: int):
    """Chronological full windows, including a final window ending at cutoff."""
    length = block_length + warmup
    if len(features) < length:
        raise ValueError("blocked: insufficient rows for a complete training block and warmup")
    starts = sorted(set([*range(0, len(features) - length + 1, block_length), len(features) - length]))
    index = torch.as_tensor(starts, device=features.device)[:, None] + torch.arange(length, device=features.device)[None, :]
    mask = torch.ones_like(index, dtype=torch.bool)
    return features[index], returns[index], benchmark[index], mask, starts


def portfolio_objective(rollout: dict[str, torch.Tensor], benchmark_returns: torch.Tensor,
                        *, downside_coefficient: float, cost_bps: float, warmup: int = 0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Negative mean block log excess plus RMS negative-block shortfall.

    Benchmark is SOXX buy-and-hold from the same initially-cash block: it pays
    one fee-funded entry. Fees in the strategy are already settled at every
    trade. No additional half-L1 turnover approximation enters the objective.
    """
    valid = rollout["mask"]
    scored = valid & (torch.arange(valid.shape[-1], device=valid.device) >= warmup)
    if bool((scored.sum(-1) == 0).any()):
        raise ValueError("blocked: training block has no scored observations after warmup")
    benchmark = benchmark_returns
    if benchmark.shape != valid.shape or not bool(torch.isfinite(benchmark).all()) or bool((benchmark <= -1).any()):
        raise ValueError("blocked: invalid allocation benchmark")
    log_benchmark = torch.log1p(benchmark) * valid
    log_benchmark = log_benchmark.clone()
    log_benchmark[..., 0] -= np.log1p(_validate_cost(cost_bps))
    relative = ((torch.log(rollout["net_equity"]) - log_benchmark) * scored).sum(-1)
    downside = torch.sqrt(torch.relu(-relative).square().mean() + 1e-12)
    loss = -relative.mean() + downside_coefficient * downside
    return loss, {"mean_block_log_excess": relative.mean(), "negative_block_shortfall_rms": downside,
                  "portfolio_loss": loss, "mean_turnover": (rollout["turnover"] * scored).sum() / scored.sum()}


def fit_policy(train_features, train_returns, *, spec: AllocationSpec, seed: int,
               auxiliary: dict[str, Any] | None = None, benchmark_returns=None,
               common_feature_indices: tuple[int, ...] | None = None) -> tuple[IntegratedAllocationPolicy, dict[str, Any]]:
    """Fit chronological Qlib-prepared rows; the workflow owns label maturity.

    Auxiliary inputs are strictly training-only: features[M,C], targets[M,K],
    common_feature_indices[C], optional nonnegative sample_weight[M]. Empty
    auxiliary data reproduces the primary-only fit and is recorded explicitly.
    """
    features = _array(train_features, name="training features")
    returns = _array(train_returns, name="training returns", columns=2)
    if len(features) != len(returns) or len(features) < 2 or (returns <= -1).any():
        raise ValueError("blocked: invalid primary allocation training rows")
    benchmark = returns[:, 1] if benchmark_returns is None else np.asarray(benchmark_returns, dtype=float)
    if benchmark.shape != (len(features),) or not np.isfinite(benchmark).all() or (benchmark <= -1).any():
        raise ValueError("blocked: invalid primary benchmark return series")
    auxiliary = auxiliary or {}
    aux_features = _array(auxiliary.get("features", np.empty((0, 0))), name="auxiliary features")
    aux_targets = _array(auxiliary.get("targets", np.empty((0, 0))), name="auxiliary targets")
    if len(aux_features) != len(aux_targets):
        raise ValueError("blocked: auxiliary features and targets have different rows")
    has_aux = len(aux_features) > 0 and spec.auxiliary_coefficient > 0
    indices = tuple(auxiliary.get("common_feature_indices", common_feature_indices if common_feature_indices is not None else range(min(16, features.shape[1]))))
    if has_aux and (len(indices) != aux_features.shape[1] or aux_targets.shape[1] != 2):
        raise ValueError("blocked: auxiliary common-feature or target schema mismatch")
    sample_weight = np.asarray(auxiliary.get("sample_weight", np.ones(len(aux_features))), dtype=float)
    if sample_weight.shape != (len(aux_features),) or not np.isfinite(sample_weight).all() or (sample_weight < 0).any() or (has_aux and sample_weight.sum() <= 0):
        raise ValueError("blocked: invalid auxiliary sample weights")
    with torch.random.fork_rng(devices=[]), _cpu_threads(spec.torch_threads):
        torch.manual_seed(int(seed))
        model = IntegratedAllocationPolicy(features.shape[1], spec.hidden_units,
            common_feature_indices=indices, auxiliary_target_count=2).double()
        with torch.no_grad():
            model.feature_mean.copy_(torch.from_numpy(features.mean(axis=0)))
            scale = features.std(axis=0)
            model.feature_scale.copy_(torch.from_numpy(np.where(scale > 1e-12, scale, 1.)))
            if has_aux:
                normalized_weight = sample_weight / sample_weight.sum()
                target_mean = (aux_targets * normalized_weight[:, None]).sum(axis=0)
                target_scale = np.sqrt(np.square(aux_targets - target_mean).T @ normalized_weight)
                model.auxiliary_target_mean.copy_(torch.from_numpy(target_mean))
                model.auxiliary_target_scale.copy_(torch.from_numpy(np.where(target_scale > 1e-12, target_scale, 1.)))
        x, r, b, mask, block_starts = _blocks(torch.from_numpy(features), torch.from_numpy(returns), torch.from_numpy(benchmark), spec.block_length, spec.warmup_sessions)
        block_rng = np.random.default_rng(int(seed) + 104729)
        if has_aux:
            ax = torch.from_numpy(aux_features)
            ay = (torch.from_numpy(aux_targets) - model.auxiliary_target_mean) / model.auxiliary_target_scale
            aw = torch.from_numpy(sample_weight / sample_weight.sum())
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
        history = []
        model.train()
        for epoch in range(spec.epochs):
            optimizer.zero_grad(set_to_none=True)
            selected_blocks = torch.as_tensor(np.sort(block_rng.choice(len(x), size=min(spec.batch_blocks, len(x)), replace=False)))
            result = rollout_policy(model, x[selected_blocks], r[selected_blocks], cost_bps=spec.training_cost_bps, mask=mask[selected_blocks])
            loss, components = portfolio_objective(result, b[selected_blocks], downside_coefficient=spec.downside_coefficient, cost_bps=spec.training_cost_bps, warmup=spec.warmup_sessions)
            auxiliary_loss = loss.new_zeros(())
            if has_aux:
                auxiliary_loss = ((model.auxiliary_prediction(ax) - ay).square().mean(-1) * aw).sum()
                loss = loss + spec.auxiliary_coefficient * auxiliary_loss
            if not bool(torch.isfinite(loss)):
                raise ValueError("blocked: nonfinite integrated allocation objective")
            loss.backward()
            if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                raise ValueError("blocked: nonfinite integrated allocation gradient")
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip)
            optimizer.step()
            history.append({"epoch": epoch + 1, "loss": float(loss.detach()), "auxiliary_loss": float(auxiliary_loss.detach()),
                            "gradient_norm": float(gradient_norm), **{key: float(value.detach()) for key, value in components.items()}})
        model.eval()
    metadata = {"spec": asdict(spec), "seed": int(seed), "training_rows": len(features),
        "feature_count": features.shape[1], "feature_scaler_source": "primary_training_rows_only",
        "architecture": "DirectContinuousAllocationNet(feature_count+3, hidden_units); softmax three net weights",
        "holdings_source": "own_post_fee_return_marked_holdings", "training_block_length": spec.block_length,
        "training_warmup": spec.warmup_sessions, "training_block_stride": spec.block_length,
        "training_block_count": len(x), "training_block_start_rows": block_starts,
        "training_boundary_overlap": "adjacent windows share warmup; final full window may overlap scored observations to include the training cutoff",
        "training_block_sampling": "without replacement; dedicated seeded numpy generator; all blocks when fewer than batch_blocks",
        "block_boundary_state": "independent initially-cash chronological blocks; first warmup observations excluded from objective",
        "evaluation_state": "continuous own-policy holdings; no block resets", "optimizer": "AdamW",
        "training_dtype": "float64", "training_torch_num_threads": spec.torch_threads,
        "objective": "negative mean scored-block log excess versus SOXX buyhold plus RMS negative-block shortfall and optional standardized auxiliary MSE; warmup excluded",
        "cost_basis": "sum absolute actual risky purchases and sales; cash fees; exact four-sign solution",
        "auxiliary_rows": len(aux_features), "auxiliary_active": has_aux, "auxiliary_empty_fallback": bool(spec.auxiliary_coefficient > 0 and not has_aux),
        "auxiliary_common_feature_indices": list(indices), "auxiliary_target_count": 2,
        "auxiliary_target_scaler_source": "sample-weighted mature auxiliary training labels only" if has_aux else "inactive neutral buffers",
        "auxiliary_head_all_variants": True, "checkpoint_selection": "fixed final epoch; no seed or checkpoint selection",
        "training_feature_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
        "training_return_sha256": hashlib.sha256(returns.tobytes()).hexdigest(),
        "training_benchmark_sha256": hashlib.sha256(np.ascontiguousarray(benchmark).tobytes()).hexdigest(),
        "auxiliary_feature_sha256": hashlib.sha256(aux_features.tobytes()).hexdigest() if has_aux else None,
        "auxiliary_target_sha256": hashlib.sha256(aux_targets.tobytes()).hexdigest() if has_aux else None,
        "auxiliary_weight_sha256": hashlib.sha256(sample_weight.tobytes()).hexdigest() if has_aux else None,
        "history": history, "no_reinforcement_learning": True}
    model.training_metadata = metadata
    return model, metadata


__all__ = ["AllocationSpec", "IntegratedAllocationPolicy", "differentiable_settlement",
           "rollout_policy", "replay_policy", "policy_target", "portfolio_objective", "fit_policy"]
