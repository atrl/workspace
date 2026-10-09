"""Dynamic beta cap versus the frozen constant for the Stage1 allocation.

The frozen Stage1 policy hard-caps the effective beta at a constant
``action_beta`` (0.6). This module tests whether replacing that constant with a
volatility-scaled cap improves risk-adjusted outcome. The cap is mechanical and
uses only past realized volatility; the upstream raw target is unchanged.

A dynamic cap is only interesting if it improves *and is stable*. This module
reports pooled metrics and a per-year Sharpe comparison so a candidate cannot be
selected on one lucky regime.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

ANNUALIZATION_SESSIONS = 252
SCOPE = "frozen_history_diagnostic_not_promotion_evidence"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class BetaCap:
    """A beta-cap rule.

    ``kind`` is ``constant`` (the frozen baseline) or ``vol_scaled``:
    ``cap_t = clip(scale * rolling_median / realized_vol_t, floor, ceiling)``.
    """

    name: str
    kind: str
    constant: float = 0.6
    scale: float = 0.6
    vol_window: int = 60
    floor: float = 0.3
    ceiling: float = 1.5

    def validate(self):
        _require(self.kind in ("constant", "vol_scaled"), "kind must be constant or vol_scaled")
        for value in (self.constant, self.scale, self.floor, self.ceiling):
            _require(isinstance(value, Real) and np.isfinite(float(value)), "cap parameters must be finite")
        _require(isinstance(self.vol_window, int) and self.vol_window > 1, "vol_window must be > 1")
        _require(0 < self.floor <= self.ceiling, "floor must be <= ceiling and positive")
        return self


def realized_vol(soxx_returns: np.ndarray, *, window: int) -> np.ndarray:
    """Annualized trailing realized volatility of the underlying, causal."""
    values = np.asarray(soxx_returns, dtype=float)
    series = pd.Series(values).rolling(window).std(ddof=1) * np.sqrt(ANNUALIZATION_SESSIONS)
    return series.bfill().to_numpy(float)


def build_caps(rule: BetaCap, soxx_returns: np.ndarray) -> np.ndarray:
    """Return the per-session effective-beta ceiling implied by a rule."""
    rule.validate()
    if rule.kind == "constant":
        return np.full(len(soxx_returns), float(rule.constant))
    vol = realized_vol(soxx_returns, window=rule.vol_window)
    median = float(np.median(vol))
    caps = float(rule.scale) * median / vol
    return np.clip(caps, float(rule.floor), float(rule.ceiling))


def run_beta_cap(raw_targets: pd.DataFrame, asset_returns: pd.DataFrame, caps: np.ndarray,
                 *, maximum_soxl_weight: float = 0.30, minimum_cash_weight: float = 0.15,
                 cost_bps: float = 10.0) -> pd.DataFrame:
    """Re-project the raw target under a per-session beta cap and replay it."""
    from src.workflow.intraday_recovery_direct_allocation import project_legal_weights

    raw_sxl = raw_targets["soxl_weight"].to_numpy(float)
    raw_sxx = raw_targets["soxx_weight"].to_numpy(float)
    soxl_ret = asset_returns["soxl"].to_numpy(float)
    soxx_ret = asset_returns["soxx"].to_numpy(float)
    _require(len(raw_sxl) == len(caps) == len(soxl_ret), "inputs must align")
    rets = np.empty(len(raw_sxl))
    held = np.array([0.0, 0.0, 1.0])
    for i in range(len(raw_sxl)):
        raw_beta = 3.0 * raw_sxl[i] + raw_sxx[i]
        effective = min(raw_beta, float(caps[i]))
        w = project_legal_weights(raw_sxl[i], raw_sxx[i], effective,
                                  minimum_cash_weight=minimum_cash_weight,
                                  maximum_soxl_weight=maximum_soxl_weight)
        ar = np.array([soxl_ret[i], soxx_ret[i], 0.0])
        drifted = held * (1.0 + ar)
        drifted = drifted / drifted.sum()
        fee = float(np.abs(w - drifted).sum()) * float(cost_bps) / 1e4
        rets[i] = float(w @ ar) - fee
        held = w
    return pd.DataFrame({"strategy_return": rets}, index=raw_targets.index)


def pooled_metrics(replay: pd.DataFrame, benchmark: np.ndarray) -> dict:
    r = replay["strategy_return"].to_numpy(float)
    b = np.asarray(benchmark, dtype=float)
    growth = np.cumprod(1.0 + r)
    bench_growth = np.cumprod(1.0 + b)
    std = r.std(ddof=1)
    return {
        "sessions": int(len(r)),
        "return_pct": 100.0 * float(growth[-1] - 1.0),
        "benchmark_return_pct": 100.0 * float(bench_growth[-1] - 1.0),
        "sharpe": float(r.mean() / std * np.sqrt(ANNUALIZATION_SESSIONS)) if std > 0 else 0.0,
        "max_drawdown_pct": 100.0 * float((growth / np.maximum.accumulate(growth) - 1.0).min()),
        "average_cap": None,
        "scope": SCOPE,
    }


def annual_sharpe(replay: pd.DataFrame, dates: pd.Series) -> pd.Series:
    frame = pd.DataFrame({
        "strategy_return": replay["strategy_return"].to_numpy(float),
        "year": pd.to_datetime(pd.Series(dates).to_numpy()).year,
    })
    def _sharpe(values):
        v = np.asarray(values, dtype=float)
        std = v.std(ddof=1)
        return float(v.mean() / std * np.sqrt(ANNUALIZATION_SESSIONS)) if std > 0 else 0.0
    return frame.groupby("year")["strategy_return"].apply(_sharpe)
