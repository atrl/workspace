"""Original Stage1 allocator with one pre-registered rolling SOXX shortfall term."""
from __future__ import annotations

import torch

from src.model.minute_pit_allocator import MinutePitAllocator
from src.workflow.pit_constituent_auxiliary_training_allocation import _holdings_aware_turnover


WINDOW = 126


def rolling_soxx_shortfall(weights: torch.Tensor, returns: torch.Tensor, *, cost_bps: int) -> torch.Tensor:
    """Mean negative 126-session relative log return, in daily-log units."""
    if weights.ndim != 2 or weights.shape[1] != 3 or returns.shape != (len(weights), 2):
        raise ValueError("blocked: original Stage1 weight/return shape required")
    if len(weights) < WINDOW or not torch.isfinite(weights).all() or not torch.isfinite(returns).all():
        raise ValueError("blocked: incomplete finite 126-session training path")
    turnover, _ = _holdings_aware_turnover(weights, returns)
    portfolio_return = (weights[:, :2] * returns).sum(dim=1) - turnover * cost_bps / 10_000.0
    portfolio_log = torch.log1p(torch.clamp(portfolio_return, min=-0.95))
    soxx_log = torch.log1p(torch.clamp(returns[:, 1], min=-0.95))
    rolling_daily_mean = (portfolio_log - soxx_log).unfold(0, WINDOW, 1).mean(dim=1)
    return torch.relu(-rolling_daily_mean).mean()


class Stage1Rolling126ShortfallAllocator(MinutePitAllocator):
    """Retain Stage1 weights, labels, constraints and optimizer; add one loss."""

    def __init__(self, **kwargs):
        if kwargs.get("extra_features") or kwargs.get("feature_masks"):
            raise ValueError("blocked: rolling objective does not add an input feature")
        super().__init__(**kwargs)
        self.rolling_shortfall_trace: list[float] = []

    def allocation_objective(self, **kwargs):
        original, turnover, annual, components = super().allocation_objective(**kwargs)
        shortfall = rolling_soxx_shortfall(kwargs["weights"], kwargs["returns"], cost_bps=kwargs["cost_bps"])
        self.rolling_shortfall_trace.append(float(shortfall.detach()))
        return original + shortfall, turnover, annual, {**components, "rolling_126_soxx_shortfall": shortfall}

    def fit(self, dataset, *, auxiliary_dataset, **kwargs):
        result = super().fit(dataset, auxiliary_dataset=auxiliary_dataset, **kwargs)
        if len(self.rolling_shortfall_trace) != self.spec.epochs:
            raise ValueError("blocked: rolling shortfall trace does not cover all frozen epochs")
        self.audit["rolling_126_objective"] = "mean_relu_negative_126_session_relative_daily_log"
        self.audit["rolling_126_penalty_coefficient"] = 1.0
        self.audit["rolling_126_shortfall_first_epoch"] = self.rolling_shortfall_trace[0]
        self.audit["rolling_126_shortfall_last_epoch"] = self.rolling_shortfall_trace[-1]
        return result

    def state_metadata(self) -> dict:
        return {**super().state_metadata(),
                "rolling_126_objective": "mean_relu_negative_126_session_relative_daily_log",
                "rolling_126_penalty_coefficient": 1.0}
