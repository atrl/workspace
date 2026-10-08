from __future__ import annotations

from typing import Any

import numpy as np


def stationary_block_bootstrap_indices(
    *,
    observation_count: int,
    path_length: int,
    path_count: int,
    expected_block_length: int,
    seed: int,
) -> np.ndarray:
    """Generate Politis-Romano stationary-bootstrap indices."""
    if observation_count < 2:
        raise ValueError("stationary bootstrap requires at least two observations")
    if min(path_length, path_count, expected_block_length) <= 0:
        raise ValueError("path and block sizes must be positive")
    rng = np.random.default_rng(seed)
    indices = np.empty((path_count, path_length), dtype=np.int32)
    indices[:, 0] = rng.integers(0, observation_count, size=path_count)
    restart_probability = 1.0 / float(expected_block_length)
    for step in range(1, path_length):
        restart = rng.random(path_count) < restart_probability
        continuation = (indices[:, step - 1] + 1) % observation_count
        starts = rng.integers(0, observation_count, size=path_count)
        indices[:, step] = np.where(restart, starts, continuation)
    return indices


def aggregate_bootstrap_paths(
    daily_returns: np.ndarray,
    daily_features: np.ndarray,
    indices: np.ndarray,
    *,
    rebalance_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate jointly sampled daily returns into rebalance-period returns."""
    returns = np.asarray(daily_returns, dtype=np.float64)
    features = np.asarray(daily_features, dtype=np.float64)
    sampled = np.asarray(indices, dtype=np.int64)
    if returns.ndim != 2 or returns.shape[1] != 2:
        raise ValueError("daily_returns must have shape [observations, 2]")
    if features.ndim != 2 or len(features) != len(returns):
        raise ValueError("daily_features must align with daily_returns")
    if sampled.ndim != 2 or sampled.shape[1] % rebalance_days:
        raise ValueError("bootstrap path length must be divisible by rebalance_days")
    if sampled.min() < 0 or sampled.max() >= len(returns):
        raise ValueError("bootstrap indices are outside the observation range")

    path_count, path_length = sampled.shape
    step_count = path_length // rebalance_days
    selected_returns = returns[sampled].reshape(path_count, step_count, rebalance_days, 2)
    period_returns = np.expm1(np.log1p(selected_returns).sum(axis=2))
    period_features = features[sampled[:, ::rebalance_days]]
    return period_returns.astype(np.float32), period_features.astype(np.float32)


def terminal_relative_wealth_metrics(
    strategy_wealth: np.ndarray,
    benchmark_wealth: np.ndarray,
) -> dict[str, Any]:
    strategy = np.asarray(strategy_wealth, dtype=np.float64)
    benchmark = np.asarray(benchmark_wealth, dtype=np.float64)
    if strategy.shape != benchmark.shape or strategy.ndim != 1 or strategy.size == 0:
        raise ValueError("terminal wealth arrays must be non-empty and aligned")
    if np.any(strategy <= 0.0) or np.any(benchmark <= 0.0):
        raise ValueError("terminal wealth must remain positive")
    ratios = strategy / benchmark
    log_ratios = np.log(ratios)
    dispersion = float(log_ratios.std(ddof=1)) if len(log_ratios) > 1 else 0.0
    active_ir = float(log_ratios.mean() / dispersion) if dispersion > 0.0 else 0.0
    return {
        "path_count": int(len(ratios)),
        "mean_terminal_wealth_ratio": float(ratios.mean()),
        "median_terminal_wealth_ratio": float(np.median(ratios)),
        "outperformance_probability": float((ratios > 1.0).mean()),
        "active_information_ratio": active_ir,
        "terminal_ratio_p05": float(np.quantile(ratios, 0.05)),
        "terminal_ratio_p10": float(np.quantile(ratios, 0.10)),
        "terminal_ratio_p90": float(np.quantile(ratios, 0.90)),
        "terminal_ratio_p95": float(np.quantile(ratios, 0.95)),
    }


def target_relative_wealth_loss(
    strategy_wealth,
    benchmark_wealth,
    *,
    annual_excess_target: float,
    horizon_years: float,
):
    """Differentiable target-based relative terminal-wealth objective."""
    target_ratio = (1.0 + float(annual_excess_target)) ** float(horizon_years)
    ratio = strategy_wealth / benchmark_wealth.clamp_min(1e-8)
    return ((ratio - target_ratio) ** 2).mean()
