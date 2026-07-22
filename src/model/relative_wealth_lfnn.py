from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from src.objective.relative_wealth import target_relative_wealth_loss, terminal_relative_wealth_metrics


class RelativeWealthPolicy(nn.Module):
    """Global-in-time feasible policy with long-only SOXL/SOXX/cash output."""

    def __init__(self, *, market_feature_count: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.market_feature_count = int(market_feature_count)
        self.input_size = 8 + self.market_feature_count
        self.network = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.network(state), dim=-1)


@dataclass(frozen=True)
class RelativeWealthTrainingSpec:
    annual_excess_target: float
    horizon_years: float = 3.0
    cost_bps: int = 10
    hidden_size: int = 32
    epochs: int = 25
    paths_per_epoch: int = 2048
    learning_rate: float = 0.003
    patience: int = 7


def _device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def simulate_policy_paths(
    policy: RelativeWealthPolicy | Sequence[RelativeWealthPolicy],
    period_returns: torch.Tensor,
    period_features: torch.Tensor,
    *,
    cost_bps: int,
) -> dict[str, torch.Tensor]:
    if period_returns.ndim != 3 or period_returns.shape[-1] != 2:
        raise ValueError("period_returns must have shape [paths, steps, 2]")
    if period_features.ndim != 3 or period_features.shape[:2] != period_returns.shape[:2]:
        raise ValueError("period_features must align with period_returns")
    policies = [policy] if isinstance(policy, RelativeWealthPolicy) else list(policy)
    if not policies:
        raise ValueError("at least one policy is required")

    path_count, step_count, _ = period_returns.shape
    device = period_returns.device
    strategy_wealth = torch.ones(path_count, device=device)
    benchmark_wealth = torch.ones(path_count, device=device) * (1.0 - float(cost_bps) / 10_000.0)
    previous_weights = torch.zeros((path_count, 3), device=device)
    previous_weights[:, 2] = 1.0
    weights_history: list[torch.Tensor] = []
    turnover_history: list[torch.Tensor] = []
    for step in range(step_count):
        time_value = torch.full((path_count, 1), step / max(step_count - 1, 1), device=device)
        relative_log_wealth = torch.log(strategy_wealth.clamp_min(1e-8)) - torch.log(
            benchmark_wealth.clamp_min(1e-8)
        )
        relative_wealth_spread = strategy_wealth / benchmark_wealth.clamp_min(1e-8) - 1.0
        state = torch.cat(
            [
                time_value,
                torch.log(strategy_wealth.clamp_min(1e-8)).unsqueeze(1),
                torch.log(benchmark_wealth.clamp_min(1e-8)).unsqueeze(1),
                relative_log_wealth.unsqueeze(1),
                relative_wealth_spread.unsqueeze(1),
                previous_weights,
                period_features[:, step, :],
            ],
            dim=1,
        )
        weights = torch.stack([candidate(state) for candidate in policies], dim=0).mean(dim=0)
        turnover = 0.5 * torch.abs(weights - previous_weights).sum(dim=1)
        strategy_wealth = strategy_wealth * (1.0 - turnover * float(cost_bps) / 10_000.0)
        asset_gross = torch.cat(
            [1.0 + period_returns[:, step, :], torch.ones((path_count, 1), device=device)], dim=1
        )
        portfolio_gross = (weights * asset_gross).sum(dim=1).clamp_min(1e-8)
        strategy_wealth = strategy_wealth * portfolio_gross
        benchmark_wealth = benchmark_wealth * (1.0 + period_returns[:, step, 1])
        previous_weights = weights * asset_gross / portfolio_gross.unsqueeze(1)
        weights_history.append(weights)
        turnover_history.append(turnover)
    return {
        "strategy_wealth": strategy_wealth,
        "benchmark_wealth": benchmark_wealth,
        "weights": torch.stack(weights_history, dim=1),
        "turnover": torch.stack(turnover_history, dim=1),
    }


def train_relative_wealth_policy(
    *,
    training_batches: Sequence[tuple[np.ndarray, np.ndarray]],
    validation_returns: np.ndarray,
    validation_features: np.ndarray,
    spec: RelativeWealthTrainingSpec,
    seed: int,
    device: str | None = None,
) -> tuple[RelativeWealthPolicy, dict[str, Any]]:
    if not training_batches:
        raise ValueError("training batches are required")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    selected_device = _device(device)
    market_feature_count = int(training_batches[0][1].shape[-1])
    policy = RelativeWealthPolicy(
        market_feature_count=market_feature_count,
        hidden_size=spec.hidden_size,
    ).to(selected_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=spec.learning_rate)
    validation_return_tensor = torch.as_tensor(validation_returns, dtype=torch.float32, device=selected_device)
    validation_feature_tensor = torch.as_tensor(validation_features, dtype=torch.float32, device=selected_device)
    best_loss = float("inf")
    best_state = deepcopy(policy.state_dict())
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(min(spec.epochs, len(training_batches))):
        policy.train()
        batch_returns, batch_features = training_batches[epoch]
        result = simulate_policy_paths(
            policy,
            torch.as_tensor(batch_returns, dtype=torch.float32, device=selected_device),
            torch.as_tensor(batch_features, dtype=torch.float32, device=selected_device),
            cost_bps=spec.cost_bps,
        )
        loss = target_relative_wealth_loss(
            result["strategy_wealth"],
            result["benchmark_wealth"],
            annual_excess_target=spec.annual_excess_target,
            horizon_years=spec.horizon_years,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()

        policy.eval()
        with torch.no_grad():
            validation = simulate_policy_paths(
                policy,
                validation_return_tensor,
                validation_feature_tensor,
                cost_bps=spec.cost_bps,
            )
            validation_loss = float(
                target_relative_wealth_loss(
                    validation["strategy_wealth"],
                    validation["benchmark_wealth"],
                    annual_excess_target=spec.annual_excess_target,
                    horizon_years=spec.horizon_years,
                ).cpu()
            )
        history.append(
            {"epoch": float(epoch + 1), "train_loss": float(loss.detach().cpu()), "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = deepcopy(policy.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= spec.patience:
                break
    policy.load_state_dict(best_state)
    policy.eval()
    with torch.no_grad():
        validation = simulate_policy_paths(
            policy,
            validation_return_tensor,
            validation_feature_tensor,
            cost_bps=spec.cost_bps,
        )
    metrics = terminal_relative_wealth_metrics(
        validation["strategy_wealth"].cpu().numpy(),
        validation["benchmark_wealth"].cpu().numpy(),
    )
    metrics.update(
        {
            "seed": int(seed),
            "epochs_completed": int(len(history)),
            "best_validation_loss": float(best_loss),
            "avg_soxl_weight": float(validation["weights"][:, :, 0].mean().cpu()),
            "avg_soxx_weight": float(validation["weights"][:, :, 1].mean().cpu()),
            "avg_cash_weight": float(validation["weights"][:, :, 2].mean().cpu()),
            "avg_turnover": float(validation["turnover"].sum(dim=1).mean().cpu()),
            "history": history,
            "device": selected_device.type,
        }
    )
    return policy.cpu(), metrics
