"""H20 replay contract: quote-clock separation, matched holding, and unit tagging.

Two distinct things live here:

- ``validate_h20_replay_contract`` is a **validator only** (no backtest). It checks the
  clock separation, the hold cadence, the covariance structure, and that the supplied
  mean/variance/daily arrays carry the correct horizon unit tags (H5 tags fed as 20 block).
- ``replay_matched_horizon`` (in ``generic_horizon_replay``) is the real daily cash/qty/fee
  replay entry; this module does not execute it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.generic_horizon import (DAILY_RISK_UNIT, allocation_clock, mean_unit,
                                      mean_variance_unit, predictor_clock, strict_positive_int,
                                      unit_tags)


def validate_unit_tags(tags: dict, *, horizon_sessions: int) -> None:
    """The arrays must be tagged with this horizon's units; a mismatched tag is blocked."""
    horizon_sessions = strict_positive_int(horizon_sessions)
    if not isinstance(tags, dict):
        raise ValueError("blocked: unit tags are required")
    expected = unit_tags(horizon_sessions)
    for key, value in expected.items():
        if tags.get(key) != value:
            raise ValueError(f"blocked: unit tag {key} is {tags.get(key)!r}, expected {value!r} "
                             f"(a different horizon's units cannot be passed as {horizon_sessions})")


def check_covariance(matrices, *, name: str, sessions: int, atol: float, require_positive_diagonal: bool) -> None:
    array = np.asarray(matrices, dtype=float)
    if array.shape != (sessions, 2, 2) or not np.isfinite(array).all():
        raise ValueError(f"blocked: {name} must be finite per-session 2x2")
    if not np.allclose(array, np.transpose(array, (0, 2, 1)), rtol=0, atol=atol):
        raise ValueError(f"blocked: {name} must be symmetric")
    if require_positive_diagonal and (np.diagonal(array, axis1=1, axis2=2) <= 0).any():
        raise ValueError(f"blocked: {name} diagonal must be positive finite squared units")
    if (np.linalg.eigvalsh(array) < -atol).any():
        raise ValueError(f"blocked: {name} must be positive semidefinite")


def validate_h20_replay_contract(*, signal_dates, trade_dates, predicted_returns, mean_variances,
                                 daily_covariances, unit_tags: dict, horizon_sessions: int = 20,
                                 rebalance_sessions: int = 20, atol: float = 1e-10) -> dict:
    """Validator only: clocks, matched holding, covariance structure and unit tagging."""
    horizon_sessions = strict_positive_int(horizon_sessions)
    rebalance_sessions = strict_positive_int(rebalance_sessions, name="rebalance_sessions")
    if rebalance_sessions != horizon_sessions:
        raise ValueError("blocked: matched holding requires rebalance_sessions == horizon_sessions")
    validate_unit_tags(unit_tags, horizon_sessions=horizon_sessions)
    signal = pd.DatetimeIndex(pd.to_datetime(signal_dates))
    trade = pd.DatetimeIndex(pd.to_datetime(trade_dates))
    if len(signal) != len(trade):
        raise ValueError("blocked: signal/trade length mismatch")
    prediction = predictor_clock(signal)
    allocation = allocation_clock(trade)
    # the allocation quote clock must be strictly after the predictor clock (never prior 16:15)
    if not bool((allocation > prediction).all()):
        raise ValueError("blocked: allocation quote clock must be after the predictor clock (never prior 16:15)")

    returns = np.asarray(predicted_returns, dtype=float)
    if returns.shape != (len(signal), 2) or not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("blocked: predicted mean returns must be finite simple returns in (-1, inf)")
    # mean-error covariance: finite/symmetric/PSD only; no imposed variance floor (the frozen
    # parent protocol defines no explicit mean-variance floor, so legal zero PSD is allowed).
    check_covariance(mean_variances, name="mean error covariance", sessions=len(signal), atol=atol,
                     require_positive_diagonal=False)
    check_covariance(daily_covariances, name="daily covariance", sessions=len(signal), atol=atol,
                     require_positive_diagonal=False)
    return {
        "kind": "validator_only",
        "horizon_sessions": horizon_sessions,
        "rebalance_sessions": rebalance_sessions,
        "matched_holding": True,
        "predictor_clock": "signal_day_16:15_America/New_York",
        "allocation_clock": "next_open_after_09:31_America/New_York",
        "allocation_after_prediction": True,
        "unit_tags_validated": True,
        "mean_unit": mean_unit(horizon_sessions),
        "mean_variance_unit": mean_variance_unit(horizon_sessions),
        "daily_risk_unit": DAILY_RISK_UNIT,
        "sessions": int(len(signal)),
    }
