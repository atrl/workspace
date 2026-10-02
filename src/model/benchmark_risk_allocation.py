"""Benchmark-risk allocation parameterization inside training and inference.

The sealed base network still reads only market features and actual holdings.
Three additional, causally estimated covariance entries are consumed solely by
the feasible-weight map. The raw risky softmax mass is a learned fraction of
the maximum feasible exposure for its composition, not a capital weight that
is clipped after training. No lower exposure, cash floor or stop clock exists.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any

import numpy as np
import torch
from torch import nn

from src.model.integrated_allocation import (
    AllocationSpec, IntegratedAllocationPolicy, _array, _blocks, _cpu_threads,
    _validate_weights, policy_target, portfolio_objective, replay_policy,
    rollout_policy,
)


COVARIANCE_FIELDS = ("cov_soxl_soxl", "cov_soxl_soxx", "cov_soxx_soxx")
RISK_PARAMETERIZATION = "risk_budget_fraction_parameterization"


def _validate_covariance(covariance: torch.Tensor) -> None:
    if covariance.shape[-1:] != (3,) or not covariance.is_floating_point() or not bool(torch.isfinite(covariance).all()):
        raise ValueError("blocked: covariance requires three finite floating entries")
    first, cross, second = covariance.unbind(-1)
    if bool((first < 0).any()) or bool((second < 0).any()):
        raise ValueError("blocked: negative benchmark covariance variance")
    product = first * second
    tolerance = 128 * torch.finfo(covariance.dtype).eps * torch.maximum(product, cross.square())
    if bool((cross.square() > product + tolerance).any()):
        raise ValueError("blocked: benchmark covariance is not positive semidefinite")


def risk_budget_components(raw_weights: torch.Tensor, covariance: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map softmax risk-budget fraction and composition to a feasible simplex.

    Covariance order is var(SOXL), cov(SOXL,SOXX), var(SOXX). All values are
    return variances in the same units. Zero variance uses exact safe branches:
    no arbitrary epsilon creates a risk budget where the benchmark has none.
    """
    _validate_weights(raw_weights, "raw allocation simplex")
    _validate_covariance(covariance)
    if covariance.shape != raw_weights.shape or covariance.dtype != raw_weights.dtype or covariance.device != raw_weights.device:
        raise ValueError("blocked: allocation and covariance shape, dtype or device differs")
    risky = raw_weights[..., :2]
    fraction = risky.sum(-1)
    safe_fraction = torch.where(fraction > 0, fraction, torch.ones_like(fraction))
    composition = risky / safe_fraction[..., None]
    first, cross, benchmark_variance = covariance.unbind(-1)
    full_variance = (composition[..., 0].square() * first
                     + 2 * composition[..., 0] * composition[..., 1] * cross
                     + composition[..., 1].square() * benchmark_variance).clamp_min(0)
    constrained = full_variance > benchmark_variance
    denominator = torch.where(constrained, full_variance, torch.ones_like(full_variance))
    positive_benchmark = benchmark_variance > 0
    benchmark_root = torch.where(positive_benchmark,
        torch.sqrt(torch.where(positive_benchmark, benchmark_variance, torch.ones_like(benchmark_variance))),
        torch.zeros_like(benchmark_variance))
    feasible_fraction = torch.where(constrained, benchmark_root / torch.sqrt(denominator), torch.ones_like(fraction))
    final_risky = feasible_fraction[..., None] * risky
    weights = torch.cat((final_risky, (1 - final_risky.sum(-1))[..., None]), dim=-1)
    portfolio_variance = (final_risky[..., 0].square() * first
                          + 2 * final_risky[..., 0] * final_risky[..., 1] * cross
                          + final_risky[..., 1].square() * benchmark_variance).clamp_min(0)
    return {"weights": weights, "learned_risk_fraction": fraction,
            "maximum_composition_exposure": feasible_fraction, "composition_variance": full_variance,
            "portfolio_variance": portfolio_variance, "benchmark_variance": benchmark_variance,
            "composition_budget_binds": constrained}


def parameterize_risk_budget(raw_weights: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    return risk_budget_components(raw_weights, covariance)["weights"]


class BenchmarkRiskAllocationPolicy(nn.Module):
    """Keep the base network intact; covariance never enters its features."""
    def __init__(self, base_feature_count: int, hidden_units: int = 8, *,
                 common_feature_indices: tuple[int, ...] = (), auxiliary_target_count: int = 2,
                 market_conditioning: bool = True):
        super().__init__()
        self.base_feature_count = int(base_feature_count)
        self.feature_count = self.base_feature_count + 3
        self.base_policy = IntegratedAllocationPolicy(base_feature_count, hidden_units,
            common_feature_indices=common_feature_indices, auxiliary_target_count=auxiliary_target_count)
        self.risk_parameterization = RISK_PARAMETERIZATION
        self.market_conditioning = bool(market_conditioning)

    def _raw_target(self, features: torch.Tensor, pretrade_weights: torch.Tensor) -> torch.Tensor:
        if self.market_conditioning:
            return self.base_policy(features[..., :self.base_feature_count], pretrade_weights)
        # Literal zero NN inputs, after any scaler: the control learns only
        # constant risk-budget fraction and composition, not market or holdings.
        zeros = features.new_zeros((*features.shape[:-1], self.base_feature_count + 3))
        return torch.softmax(self.base_policy.allocator(zeros), dim=-1)

    def forward(self, features: torch.Tensor, pretrade_weights: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_count:
            raise ValueError("blocked: risk-budget policy requires base features plus three covariance entries")
        raw = self._raw_target(features, pretrade_weights)
        return parameterize_risk_budget(raw, features[..., self.base_feature_count:])

    def auxiliary_prediction(self, common_features: torch.Tensor) -> torch.Tensor:
        return self.base_policy.auxiliary_prediction(common_features)

    def predict_target(self, features, pretrade_weights) -> np.ndarray:
        return policy_target(self, features, pretrade_weights)

    def risk_diagnostics(self, features: torch.Tensor, pretrade_weights: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self._raw_target(features, pretrade_weights)
        return risk_budget_components(raw, features[..., self.base_feature_count:])


def fit_policy(train_features, train_returns, *, spec: AllocationSpec, seed: int,
               auxiliary: dict[str, Any] | None = None, benchmark_returns=None,
               common_feature_indices: tuple[int, ...] | None = None,
               market_conditioning: bool = True) -> tuple[BenchmarkRiskAllocationPolicy, dict[str, Any]]:
    """Reuse sealed objective, blocks and accounting with the in-model map.

    The Qlib workflow must supply trailing covariance, mature outcomes and PIT
    auxiliary rows. Input order is the unchanged base feature schema followed
    by ``COVARIANCE_FIELDS``. Only the base columns are standardized for the NN.
    """
    features = _array(train_features, name="risk-budget training features")
    returns = _array(train_returns, name="risk-budget training returns", columns=2)
    if features.shape[1] <= 3 or len(features) != len(returns) or len(features) < 2 or (returns <= -1).any():
        raise ValueError("blocked: invalid primary risk-budget training inputs")
    _validate_covariance(torch.from_numpy(features[:, -3:]))
    base_features = features[:, :-3]
    benchmark = returns[:, 1] if benchmark_returns is None else np.asarray(benchmark_returns, dtype=float)
    if benchmark.shape != (len(features),) or not np.isfinite(benchmark).all() or (benchmark <= -1).any():
        raise ValueError("blocked: invalid risk-budget benchmark returns")
    auxiliary = auxiliary or {}
    aux_features = _array(auxiliary.get("features", np.empty((0, 0))), name="auxiliary features")
    aux_targets = _array(auxiliary.get("targets", np.empty((0, 0))), name="auxiliary targets")
    if len(aux_features) != len(aux_targets):
        raise ValueError("blocked: auxiliary row count differs")
    has_aux = len(aux_features) > 0 and spec.auxiliary_coefficient > 0
    if not market_conditioning and spec.auxiliary_coefficient > 0:
        raise ValueError("blocked: no-signal control cannot use auxiliary learning")
    indices = tuple(auxiliary.get("common_feature_indices", common_feature_indices if common_feature_indices is not None else range(min(16, base_features.shape[1]))))
    if has_aux and (len(indices) != aux_features.shape[1] or aux_targets.shape[1] != 2):
        raise ValueError("blocked: auxiliary schema differs from the shared base projection")
    sample_weight = np.asarray(auxiliary.get("sample_weight", np.ones(len(aux_features))), dtype=float)
    if sample_weight.shape != (len(aux_features),) or not np.isfinite(sample_weight).all() or (sample_weight < 0).any() or (has_aux and sample_weight.sum() <= 0):
        raise ValueError("blocked: invalid auxiliary sample weights")
    with torch.random.fork_rng(devices=[]), _cpu_threads(spec.torch_threads):
        torch.manual_seed(int(seed))
        model = BenchmarkRiskAllocationPolicy(base_features.shape[1], spec.hidden_units,
            common_feature_indices=indices, auxiliary_target_count=2, market_conditioning=market_conditioning).double()
        with torch.no_grad():
            model.base_policy.feature_mean.copy_(torch.from_numpy(base_features.mean(axis=0)))
            scale = base_features.std(axis=0)
            model.base_policy.feature_scale.copy_(torch.from_numpy(np.where(scale > 1e-12, scale, 1.)))
            if has_aux:
                normalized_weight = sample_weight / sample_weight.sum()
                target_mean = (aux_targets * normalized_weight[:, None]).sum(axis=0)
                target_scale = np.sqrt(np.square(aux_targets - target_mean).T @ normalized_weight)
                model.base_policy.auxiliary_target_mean.copy_(torch.from_numpy(target_mean))
                model.base_policy.auxiliary_target_scale.copy_(torch.from_numpy(np.where(target_scale > 1e-12, target_scale, 1.)))
        x, r, b, mask, starts = _blocks(torch.from_numpy(features), torch.from_numpy(returns),
            torch.from_numpy(benchmark), spec.block_length, spec.warmup_sessions)
        block_rng = np.random.default_rng(int(seed) + 104729)
        if has_aux:
            ax = torch.from_numpy(aux_features)
            ay = (torch.from_numpy(aux_targets) - model.base_policy.auxiliary_target_mean) / model.base_policy.auxiliary_target_scale
            aw = torch.from_numpy(sample_weight / sample_weight.sum())
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
        history = []
        model.train()
        for epoch in range(spec.epochs):
            optimizer.zero_grad(set_to_none=True)
            selected_blocks = torch.as_tensor(np.sort(block_rng.choice(len(x), size=min(spec.batch_blocks, len(x)), replace=False)))
            result = rollout_policy(model, x[selected_blocks], r[selected_blocks], cost_bps=spec.training_cost_bps, mask=mask[selected_blocks])
            loss, components = portfolio_objective(result, b[selected_blocks], downside_coefficient=spec.downside_coefficient,
                cost_bps=spec.training_cost_bps, warmup=spec.warmup_sessions)
            auxiliary_loss = loss.new_zeros(())
            if has_aux:
                auxiliary_loss = ((model.auxiliary_prediction(ax) - ay).square().mean(-1) * aw).sum()
                loss = loss + spec.auxiliary_coefficient * auxiliary_loss
            if not bool(torch.isfinite(loss)):
                raise ValueError("blocked: nonfinite benchmark-risk allocation objective")
            loss.backward()
            if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                raise ValueError("blocked: nonfinite benchmark-risk allocation gradient")
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip)
            optimizer.step()
            history.append({"epoch": epoch + 1, "loss": float(loss.detach()), "auxiliary_loss": float(auxiliary_loss.detach()),
                "gradient_norm": float(gradient_norm), **{key: float(value.detach()) for key, value in components.items()}})
        model.eval()
    metadata = {"spec": asdict(spec), "seed": int(seed), "training_rows": len(features),
        "feature_count": features.shape[1], "network_market_feature_count": base_features.shape[1],
        "market_conditioning": bool(market_conditioning),
        "neural_input_contract": "market features and actual holdings" if market_conditioning else "literal zero market and holding inputs; covariance only affects feasible-weight parameterization",
        "covariance_fields": list(COVARIANCE_FIELDS), "covariance_enters_network": False,
        "feature_scaler_source": "primary training base features only; covariance never standardized",
        "architecture": "unchanged IntegratedAllocationPolicy plus benchmark-risk fraction parameterization",
        "risk_parameterization": RISK_PARAMETERIZATION,
        "risk_budget": "SOXX variance from supplied causal expanding covariance; ceiling only, no minimum risk exposure",
        "raw_softmax_interpretation": "risky mass is fraction of maximum feasible risky exposure for its composition",
        "holdings_source": "own post-fee return-marked holdings after in-model risk-budget allocation",
        "training_block_length": spec.block_length, "training_warmup": spec.warmup_sessions,
        "training_block_stride": spec.block_length, "training_block_count": len(x), "training_block_start_rows": starts,
        "training_block_sampling": "without replacement; dedicated seeded generator; all when fewer than batch_blocks",
        "training_boundary_overlap": "adjacent windows share warmup; final full window may overlap scored observations",
        "block_boundary_state": "initially cash; warmup excluded from objective", "evaluation_state": "continuous own holdings",
        "optimizer": "AdamW", "training_dtype": "float64", "training_torch_num_threads": spec.torch_threads,
        "objective": "sealed integrated-allocation scored-block log-excess plus RMS negative-block shortfall and optional weighted auxiliary MSE",
        "cost_basis": "sealed exact four-sign cash-funded risky purchases and sales",
        "auxiliary_rows": len(aux_features), "auxiliary_active": has_aux,
        "auxiliary_empty_fallback": bool(spec.auxiliary_coefficient > 0 and not has_aux),
        "auxiliary_common_feature_indices": list(indices), "auxiliary_target_count": 2, "auxiliary_head_all_variants": True,
        "auxiliary_target_scaler_source": "sample-weighted mature auxiliary training labels only" if has_aux else "inactive neutral buffers",
        "checkpoint_selection": "fixed final epoch; no seed or checkpoint selection",
        "training_feature_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
        "training_return_sha256": hashlib.sha256(returns.tobytes()).hexdigest(),
        "training_benchmark_sha256": hashlib.sha256(np.ascontiguousarray(benchmark).tobytes()).hexdigest(),
        "auxiliary_feature_sha256": hashlib.sha256(aux_features.tobytes()).hexdigest() if has_aux else None,
        "auxiliary_target_sha256": hashlib.sha256(aux_targets.tobytes()).hexdigest() if has_aux else None,
        "auxiliary_weight_sha256": hashlib.sha256(sample_weight.tobytes()).hexdigest() if has_aux else None,
        "history": history, "no_reinforcement_learning": True}
    model.training_metadata = metadata
    return model, metadata


__all__ = ["AllocationSpec", "BenchmarkRiskAllocationPolicy", "COVARIANCE_FIELDS", "RISK_PARAMETERIZATION",
           "parameterize_risk_budget", "risk_budget_components", "fit_policy", "policy_target", "rollout_policy", "replay_policy"]
