"""Causal intraday early-session momentum policy for SOXL/SOXX/cash.

The opening 60 minutes (09:31 through 10:30) are observable before the remaining
session. This module tests whether that observed path, with no return forecast,
supports a long-only allocation for the 10:30-to-close window that beats SOXX
buy-and-hold on return, drawdown and Sharpe together.

Every decision uses state known at 10:30 and is applied to that session's
10:30-to-close return. There is no look-ahead, no overnight position, and no
fitted parameter inside this module. Callers supply a bounded, pre-registered
threshold family and own segmentation, selection and cost handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

ANNUALIZATION_SESSIONS = 252
ASSETS = ("SOXL", "SOXX", "cash")
SCOPE = "reused_history_and_forward_diagnostic_not_production_evidence"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class MomentumPolicy:
    """A pre-registered early-momentum exposure rule.

    ``long_threshold``: hold SOXL for the rest of the session when the early
    return exceeds it. ``short_threshold``: hold cash when the early return falls
    below the negative of it (risk-off). Otherwise hold SOXX. Both thresholds are
    nonnegative; ``short_threshold=1.0`` effectively disables the cash branch.
    """

    name: str
    long_threshold: float
    short_threshold: float = 1.0
    signal: str = "soxl_early"

    def validate(self):
        _require(isinstance(self.name, str) and self.name, "policy name is required")
        _require(self.signal in ("soxl_early", "soxx_early"), "signal must be soxl_early or soxx_early")
        for value, label in ((self.long_threshold, "long_threshold"),
                             (self.short_threshold, "short_threshold")):
            _require(isinstance(value, Real) and not isinstance(value, bool)
                     and np.isfinite(value) and value >= 0, label + " must be a finite nonnegative real")
        return self


@dataclass(frozen=True)
class EarlySessionFrame:
    """One row per session with the causally observable early path and close path."""

    frame: pd.DataFrame

    @property
    def signal_date(self) -> pd.Series:
        return self.frame["signal_date"]


def build_early_session_frame(soxl: pd.DataFrame, soxx: pd.DataFrame, *,
                              first_bar_time: str = "09:45", decision_time: str = "10:30",
                              close_time: str = "16:00") -> EarlySessionFrame:
    """Assemble per-session early return (session open to 10:30) and rest-of-day returns.

    ``soxl``/``soxx`` are intraday frames indexed by a tz-aware datetime with
    ``open`` and ``close`` columns labeled by bar end time. The first bar's open
    is the session open; the early return is the 10:30 close over that open. The
    rest-of-day return is the 16:00 close over the 10:30 close. Both SOXL and SOXX
    rest returns are retained so either asset can be selected at 10:30.
    """
    for name, frame in (("soxl", soxl), ("soxx", soxx)):
        _require(isinstance(frame, pd.DataFrame) and {"open", "close"}.issubset(frame.columns),
                 f"{name} intraday frame needs open and close columns")
        _require(isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None,
                 f"{name} intraday index must be a tz-aware DatetimeIndex")
    open_clock = pd.Timestamp(first_bar_time).time()
    decision_clock = pd.Timestamp(decision_time).time()
    close_clock = pd.Timestamp(close_time).time()

    def picks(frame):
        index = frame.index
        day = index.normalize()
        opening = frame.loc[index.time == open_clock, "open"].set_axis(day[index.time == open_clock])
        decision = frame.loc[index.time == decision_clock, "close"].set_axis(day[index.time == decision_clock])
        closing = frame.loc[index.time == close_clock, "close"].set_axis(day[index.time == close_clock])
        return opening, decision, closing

    open_x, decision_x, close_x = picks(soxl)
    open_y, decision_y, close_y = picks(soxx)
    frame = pd.DataFrame({
        "soxl_open": open_x, "soxl_decision": decision_x, "soxl_close": close_x,
        "soxx_open": open_y, "soxx_decision": decision_y, "soxx_close": close_y,
    }).dropna()
    frame = frame[frame.index.dayofweek < 5]
    frame.index.name = "signal_date"
    frame = frame.reset_index()
    frame["soxl_early"] = frame["soxl_decision"] / frame["soxl_open"] - 1.0
    frame["soxx_early"] = frame["soxx_decision"] / frame["soxx_open"] - 1.0
    frame["early_return"] = frame["soxl_early"]
    frame["rest_soxl"] = frame["soxl_close"] / frame["soxl_decision"] - 1.0
    frame["rest_soxx"] = frame["soxx_close"] / frame["soxx_decision"] - 1.0
    _require(len(frame) > 0, "no aligned early-session rows were built")
    return EarlySessionFrame(frame=frame)


def target_weights(early_return: float, policy: MomentumPolicy) -> np.ndarray:
    """Map the observed early return to a funded (SOXL, SOXX, cash) target."""
    policy.validate()
    value = float(early_return)
    _require(np.isfinite(value), "early return must be finite")
    if value <= -policy.short_threshold:
        return np.array([0.0, 0.0, 1.0])
    if value >= policy.long_threshold:
        return np.array([1.0, 0.0, 0.0])
    return np.array([0.0, 1.0, 0.0])


def simulate_policy(frame: pd.DataFrame, policy: MomentumPolicy, *, cost_bps: float) -> pd.DataFrame:
    """Replay the 10:30 decision against the same-session rest-of-day returns.

    Turnover at 10:30 relative to the prior session's 10:30 target is charged at
    the decision cost. No overnight position exists, so the entry cost each
    session is bounded by the target change.
    """
    _require(isinstance(cost_bps, Real) and not isinstance(cost_bps, bool)
             and np.isfinite(cost_bps) and cost_bps >= 0, "cost_bps must be finite and nonnegative")
    ordered = frame.sort_values("signal_date").reset_index(drop=True)
    prior = np.array([0.0, 1.0, 0.0])  # neutral: benchmark position
    fee_rate = float(cost_bps) / 10_000.0
    records = []
    for row in ordered.itertuples(index=False):
        signal_value = getattr(row, policy.signal)
        target = target_weights(signal_value, policy)
        turnover = float(np.abs(target - prior).sum())
        gross = target[0] * float(row.rest_soxl) + target[1] * float(row.rest_soxx)
        records.append({
            "signal_date": row.signal_date,
            "soxl_weight": target[0], "soxx_weight": target[1], "cash_weight": target[2],
            "turnover": turnover, "gross_return": gross,
            "strategy_return": gross - turnover * fee_rate,
            "rest_soxl": float(row.rest_soxl), "rest_soxx": float(row.rest_soxx),
        })
        prior = target
    result = pd.DataFrame(records)
    _require(not result.empty, "policy simulation produced no sessions")
    return result


def benchmark_replay(frame: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    """SOXX buy-and-hold over the identical 10:30-to-close window."""
    _require(isinstance(frame, pd.DataFrame) and not frame.empty, "frame is required")
    ordered = frame.sort_values("signal_date").reset_index(drop=True)
    return pd.DataFrame({
        "signal_date": ordered["signal_date"].to_numpy(),
        "soxl_weight": 0.0, "soxx_weight": 1.0, "cash_weight": 0.0,
        "turnover": 0.0, "gross_return": ordered["rest_soxx"].to_numpy(float),
        "strategy_return": ordered["rest_soxx"].to_numpy(float),
        "rest_soxl": ordered["rest_soxl"].to_numpy(float),
        "rest_soxx": ordered["rest_soxx"].to_numpy(float),
    })


def annual_wealth(replay: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-year wealth for a single replay path."""
    frame = replay.copy()
    frame["year"] = pd.to_datetime(frame["signal_date"]).dt.year
    rows = []
    for year, group in frame.groupby("year"):
        rows.append({
            "year": int(year),
            "sessions": int(len(group)),
            "wealth": float(np.prod(1.0 + group["strategy_return"].to_numpy(float))),
            "average_soxl_weight": float(group["soxl_weight"].mean()),
            "average_cash_weight": float(group["cash_weight"].mean()),
            "turnover": float(group["turnover"].sum()),
        })
    return pd.DataFrame(rows)
