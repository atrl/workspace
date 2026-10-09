"""Systematic causal signal search over every available SOXL data source.

Goal: find a *stable* excess over SOXX buy-and-hold defined as simultaneously
beating return, drawdown and Sharpe, with a majority of positive calendar years.
Signals are built only from past-available information and are never fitted to
the evaluation window.

This module is deliberately descriptive. It takes a set of named causal signals,
a target return series, and reports per-signal annual and pooled statistics. The
caller owns segmentation, selection and Recorder evidence. No output grants
production eligibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

ANNUALIZATION_SESSIONS = 252
SCOPE = "reused_history_and_forward_diagnostic_not_production_evidence"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class SignalSpec:
    """A named causal signal and the exposure rule it drives.

    ``direction`` is ``+1`` for momentum (hold the risky asset when the signal is
    high) or ``-1`` for reversal (hold it when the signal is low). ``threshold``
    is the signal quantile in [0, 1] that defines the active state: for a
    momentum signal the risky asset is held when the signal is at or above the
    quantile, and for a reversal signal when it is at or below the quantile.
    """

    name: str
    signal: pd.Series
    direction: int
    threshold: float
    source: str

    def validate(self):
        _require(isinstance(self.name, str) and self.name, "signal name is required")
        _require(self.direction in (1, -1), "direction must be +1 or -1")
        _require(isinstance(self.threshold, Real) and 0.0 <= float(self.threshold) <= 1.0,
                 "threshold must be a probability in [0,1]")
        _require(isinstance(self.signal, pd.Series), "signal must be a pandas Series")
        return self


def build_exposure(signal: pd.Series, direction: int, threshold: float) -> pd.Series:
    """Return a 0/1 risky-asset exposure from a past-available signal.

    The quantile cut is computed on the expanding window available up to each
    date, so the threshold itself never uses future data.
    """
    _require(direction in (1, -1), "direction must be +1 or -1")
    _require(0.0 <= float(threshold) <= 1.0, "threshold must be a probability in [0,1]")
    values = signal.astype(float)
    expanding_cut = values.expanding(min_periods=60).quantile(threshold)
    if direction == 1:
        active = values >= expanding_cut
    else:
        active = values <= expanding_cut
    exposure = active.astype(float)
    # No classification is possible before the expanding window fills.
    exposure[values.isna() | expanding_cut.isna()] = np.nan
    return exposure


def evaluate_signal(signal, risky_return, safe_return, *, direction, threshold, cost_bps, min_history=60):
    """Replay a causal exposure rule switching between risky and safe assets.

    Exposure at day ``t`` is decided from the signal known at ``t-1`` and applied
    to day ``t`` returns. Turnover is charged on each exposure change.
    """
    _require(isinstance(cost_bps, Real) and not isinstance(cost_bps, bool) and cost_bps >= 0,
             "cost_bps must be nonnegative")
    frame = pd.DataFrame({"signal": signal.astype(float),
                          "risky": risky_return.astype(float),
                          "safe": safe_return.astype(float)}).dropna()
    if len(frame) <= min_history + 5:
        return None
    exposure = build_exposure(frame["signal"], direction, threshold)
    applied = exposure.shift(1).fillna(0.0)
    turnover = (applied - applied.shift(1).fillna(0.0)).abs()
    strategy = applied * frame["risky"] + (1.0 - applied) * frame["safe"] - turnover * float(cost_bps) / 1e4
    bench = frame["safe"]
    return {"strategy": strategy, "benchmark": bench, "exposure": applied,
            "turnover": turnover, "sessions": int(len(frame))}


def _metrics(returns: pd.Series) -> dict:
    values = returns.to_numpy(float)
    growth = np.cumprod(1.0 + values)
    years = len(values) / ANNUALIZATION_SESSIONS
    running_max = np.maximum.accumulate(growth)
    drawdown = growth / running_max - 1.0
    std = values.std(ddof=1)
    return {
        "terminal_wealth": float(growth[-1]),
        "cagr_pct": 100.0 * (float(growth[-1]) ** (1.0 / years) - 1.0) if years > 0 else 0.0,
        "sharpe": float(values.mean() / std * np.sqrt(ANNUALIZATION_SESSIONS)) if std > 0 else 0.0,
        "max_drawdown_pct": 100.0 * float(drawdown.min()),
        "worst_day_pct": 100.0 * float(values.min()),
    }


def summarize_search(results: dict[str, dict], *, minimum_positive_years: int) -> pd.DataFrame:
    """Pooled and per-year comparison of every evaluated signal against the benchmark."""
    rows = []
    for name, replay in results.items():
        strategy, benchmark = replay["strategy"], replay["benchmark"]
        sm, bm = _metrics(strategy), _metrics(benchmark)
        annual = pd.DataFrame({"s": strategy, "b": benchmark})
        annual["year"] = pd.to_datetime(annual.index).year
        per_year = annual.groupby("year").apply(
            lambda g: float(np.prod(1.0 + g["s"]) / np.prod(1.0 + g["b"])), include_groups=False)
        positive_years = int((per_year > 1.0).sum())
        rows.append({
            "signal": name,
            "sessions": replay["sessions"],
            "strategy_return_pct": 100.0 * (sm["terminal_wealth"] - 1.0),
            "benchmark_return_pct": 100.0 * (bm["terminal_wealth"] - 1.0),
            "wealth_ratio_vs_soxx": sm["terminal_wealth"] / bm["terminal_wealth"],
            "strategy_sharpe": sm["sharpe"],
            "benchmark_sharpe": bm["sharpe"],
            "sharpe_difference": sm["sharpe"] - bm["sharpe"],
            "strategy_max_drawdown_pct": sm["max_drawdown_pct"],
            "benchmark_max_drawdown_pct": bm["max_drawdown_pct"],
            "max_drawdown_improvement_pct": bm["max_drawdown_pct"] - sm["max_drawdown_pct"],
            "average_exposure": float(replay["exposure"].mean()),
            "total_turnover": float(replay["turnover"].sum()),
            "positive_year_count": positive_years,
            "year_count": int(per_year.size),
            "beats_return": sm["terminal_wealth"] > bm["terminal_wealth"],
            "beats_sharpe": sm["sharpe"] > bm["sharpe"],
            "beats_drawdown": sm["max_drawdown_pct"] > bm["max_drawdown_pct"],
            "stable_excess": bool(sm["terminal_wealth"] > bm["terminal_wealth"]
                                  and sm["sharpe"] > bm["sharpe"]
                                  and sm["max_drawdown_pct"] > bm["max_drawdown_pct"]
                                  and positive_years >= minimum_positive_years),
        })
    return pd.DataFrame(rows).sort_values(
        ["stable_excess", "sharpe_difference"], ascending=[False, False]).reset_index(drop=True)


def file_digest(path) -> str:
    import hashlib
    from pathlib import Path
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
