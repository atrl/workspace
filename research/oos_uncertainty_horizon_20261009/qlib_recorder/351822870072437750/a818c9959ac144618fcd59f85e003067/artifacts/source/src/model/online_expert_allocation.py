"""Causal fixed-share aggregation for caller-defined allocation experts.

The Qlib workflow owns data, cost-aware expert settlement, and the recorder.
This module only consumes already available *net* returns. It does not fetch
data, fit a model, choose experts, or search hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np


def _simplex(values, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if (result.ndim != 1 or result.size == 0 or not np.isfinite(result).all()
            or (result < 0).any() or not np.isclose(result.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError(f"blocked: {name} must be a finite nonnegative simplex")
    # Only remove floating-point summation error after validating the contract.
    return result / result.sum()


def _aware_utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"blocked: {name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class FixedShareConfig:
    """All research choices are explicit; rows follow ``expert_names`` order.

    Each expert row and the anchor use the same caller-defined asset order.
    ``share`` mixes the updated expert distribution back toward ``prior``;
    ``active_fraction`` mixes its allocation with the fixed anchor. Zero prior
    mass disables an expert, including after the fixed-share step.
    """

    expert_names: tuple[str, ...]
    expert_weights: tuple[tuple[float, ...], ...]
    anchor_weights: tuple[float, ...]
    prior: tuple[float, ...]
    active_fraction: float
    eta: float
    share: float

    def __post_init__(self) -> None:
        names = tuple(self.expert_names)
        if (not names or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("blocked: expert_names must be nonempty unique strings")
        anchor = _simplex(self.anchor_weights, name="anchor_weights")
        prior = _simplex(self.prior, name="prior")
        experts = np.asarray(self.expert_weights, dtype=float)
        if experts.shape != (len(names), len(anchor)) or prior.shape != (len(names),):
            raise ValueError("blocked: expert, anchor and prior dimensions differ")
        experts = np.vstack([_simplex(row, name="expert_weights row") for row in experts])
        for name in ("active_fraction", "share", "eta"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim != 0 or not np.isfinite(value) or value < 0:
                raise ValueError(f"blocked: {name} must be a finite nonnegative scalar")
            if name != "eta" and value > 1:
                raise ValueError(f"blocked: {name} must be in [0, 1]")
            object.__setattr__(self, name, float(value))
        # Canonical immutable values also protect against mutable input arrays.
        object.__setattr__(self, "expert_names", names)
        object.__setattr__(self, "expert_weights", tuple(tuple(row) for row in experts))
        object.__setattr__(self, "anchor_weights", tuple(anchor))
        object.__setattr__(self, "prior", tuple(prior))


class FixedShareAllocator:
    """Deterministic exponential weights using strictly mature feedback only.

    For each accepted observation, the update is proportional to
    ``posterior * exp(eta * (log1p(expert_net_returns) - log1p(benchmark)))``,
    followed by ``(1-share) * updated + share * prior``. Returns must be from
    consistently settled expert and benchmark paths, supplied by the caller.

    Availability and decision times must be timezone-aware. An observation is
    usable only when ``available_at < decision_at``. Availability must never
    go backwards, decisions must never go backwards, and IDs cannot repeat.
    Several distinct observations may mature together or be consumed at the
    same decision. Call ``update`` for those observations, then ``target``.
    """

    def __init__(self, config: FixedShareConfig):
        if not isinstance(config, FixedShareConfig):
            raise TypeError("config must be a FixedShareConfig")
        self._config = config
        self._prior = np.asarray(config.prior, dtype=float)
        self._experts = np.asarray(config.expert_weights, dtype=float)
        self._anchor = np.asarray(config.anchor_weights, dtype=float)
        with np.errstate(divide="ignore"):
            self._log_posterior = np.log(self._prior)
        self._seen_ids: set[str] = set()
        self._last_available_at: datetime | None = None
        self._last_decision_at: datetime | None = None

    @property
    def config(self) -> FixedShareConfig:
        return self._config

    @property
    def posterior(self) -> np.ndarray:
        """Return a fresh expert-weight vector; callers cannot alter state."""
        weights = np.exp(self._log_posterior)
        return weights / weights.sum()

    @property
    def observation_count(self) -> int:
        return len(self._seen_ids)

    @property
    def last_available_at(self) -> datetime | None:
        """UTC availability time of the last consumed observation."""
        return self._last_available_at

    @property
    def last_decision_at(self) -> datetime | None:
        return self._last_decision_at

    def _decision_time(self, decision_at: datetime) -> datetime:
        decision = _aware_utc(decision_at, name="decision_at")
        if self._last_decision_at is not None and decision < self._last_decision_at:
            raise ValueError("blocked: decision_at cannot go backwards")
        return decision

    def update(self, observation_id: str, available_at: datetime, decision_at: datetime,
               expert_net_returns: Sequence[float], benchmark_net_return: float) -> np.ndarray:
        """Consume one mature observation atomically and return the posterior."""
        if not isinstance(observation_id, str) or not observation_id.strip():
            raise ValueError("blocked: observation_id must be a nonempty string")
        if observation_id in self._seen_ids:
            raise ValueError("blocked: duplicate observation_id")
        available = _aware_utc(available_at, name="available_at")
        decision = self._decision_time(decision_at)
        if available >= decision:
            raise ValueError("blocked: feedback is not strictly mature at decision_at")
        if self._last_available_at is not None and available < self._last_available_at:
            raise ValueError("blocked: available_at cannot go backwards")
        returns = np.asarray(expert_net_returns, dtype=float)
        benchmark = np.asarray(benchmark_net_return, dtype=float)
        if (returns.shape != self._prior.shape or not np.isfinite(returns).all()
                or (returns <= -1).any()):
            raise ValueError("blocked: expert_net_returns must be finite, aligned and greater than -1")
        if benchmark.ndim != 0 or not np.isfinite(benchmark) or benchmark <= -1:
            raise ValueError("blocked: benchmark_net_return must be finite and greater than -1")

        log_excess = np.log1p(returns) - np.log1p(float(benchmark))
        # Subtract a shared reward constant before multiplication. The best
        # supported expert remains finite even for a very large finite eta.
        supported = np.isfinite(self._log_posterior)
        centered = log_excess[supported] - log_excess[supported].max()
        logits = np.full_like(self._log_posterior, -np.inf)
        with np.errstate(over="ignore", under="ignore"):
            logits[supported] = self._log_posterior[supported] + self.config.eta * centered
            logits -= logits.max()
            log_updated = logits - np.log(np.exp(logits).sum())
        if self.config.share == 0:
            log_next = log_updated
        elif self.config.share == 1:
            with np.errstate(divide="ignore"):
                log_next = np.log(self._prior)
        else:
            with np.errstate(divide="ignore", under="ignore"):
                log_next = np.logaddexp(np.log1p(-self.config.share) + log_updated,
                                       np.log(self.config.share) + np.log(self._prior))
        if np.isnan(log_next).any() or not np.isfinite(log_next).any():
            raise ValueError("blocked: fixed-share posterior is numerically invalid")
        # No mutation occurs until all validation and calculation succeeds.
        self._log_posterior = log_next
        self._seen_ids.add(observation_id)
        self._last_available_at = available
        self._last_decision_at = decision
        return self.posterior

    def target(self, decision_at: datetime) -> np.ndarray:
        """Blend the current expert allocation with the caller's fixed anchor."""
        decision = self._decision_time(decision_at)
        target = ((1 - self.config.active_fraction) * self._anchor
                  + self.config.active_fraction * (self.posterior @ self._experts))
        self._last_decision_at = decision
        return target
