"""Legacy local-date close-to-close diagnostic with a restricted interpretation.

Asian cash markets can close before the same-local-date US open. However, this
module's US close-to-close target starts before that Asian close, so a same-date
correlation cannot be earned over the entire target interval. The lag1 route is
a conservative test for that target; it does not rule out same-date information
for a later US open-to-close window. Frozen numerical behavior is unchanged.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

ANNUALIZATION_SESSIONS = 252
SCOPE = "new_external_data_diagnostic_not_production_evidence"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def load_manifest(manifest: dict) -> pd.DataFrame:
    """Validate and flatten a Yahoo-style series manifest into a frame."""
    _require(isinstance(manifest, dict) and "series" in manifest, "manifest needs a series map")
    rows = []
    for name, info in manifest["series"].items():
        rows.append({"series": name, "symbol": info["symbol"], "timezone": info["timezone"],
                     "rows": int(info["rows"]), "start": info["start"], "end": info["end"]})
    frame = pd.DataFrame(rows).sort_values("series").reset_index(drop=True)
    _require(not frame.empty, "manifest is empty")
    return frame


def information_coefficient(predictor: pd.Series, target: pd.Series, *, lag: int) -> dict:
    """Pearson and Spearman IC between a lagged predictor and the target.

    ``lag=0`` aligns predictor at date t with target at date t (same local date).
    ``lag=1`` uses predictor at t-1 for target at t, which is the conservative
    alignment used for this legacy close-to-close target.
    """
    _require(isinstance(lag, Integral) and lag >= 0, "lag must be a nonnegative integer")
    shifted = predictor.shift(lag)
    frame = pd.DataFrame({"x": shifted, "y": target}).dropna()
    frame = frame[(frame.index.dayofweek < 5)]
    _require(len(frame) > 60, "not enough overlapping observations")
    if frame["x"].std() == 0 or frame["y"].std() == 0:
        return {"pearson_ic": None, "spearman_ic": None, "observations": int(len(frame))}
    return {
        "pearson_ic": float(frame["x"].corr(frame["y"])),
        "spearman_ic": float(frame["x"].corr(frame["y"], method="spearman")),
        "observations": int(len(frame)),
    }


def switch_strategy(signal: pd.Series, soxl: pd.Series, soxx: pd.Series, *,
                    lag: int, cost_bps: float) -> dict:
    """Hold SOXL when the lagged signal is positive, else SOXX.

    The exposure uses the signal from ``lag`` sessions earlier so the forecast
    alignment is explicit. Turnover is charged on each exposure change.
    """
    _require(cost_bps >= 0, "cost_bps must be nonnegative")
    applied = (signal.shift(lag) > 0).astype(float)
    frame = pd.DataFrame({"exposure": applied, "soxl": soxl, "soxx": soxx}).dropna()
    frame = frame[(frame.index.dayofweek < 5)]
    _require(len(frame) > 60, "not enough sessions")
    turnover = (frame["exposure"] - frame["exposure"].shift(1).fillna(0.0)).abs()
    strategy = (frame["exposure"] * frame["soxl"] + (1 - frame["exposure"]) * frame["soxx"]
                - turnover * float(cost_bps) / 1e4)
    return {"strategy": strategy, "benchmark": frame["soxx"], "exposure": frame["exposure"],
            "turnover": turnover, "sessions": int(len(frame))}


def _metrics(returns: pd.Series) -> dict:
    values = returns.to_numpy(float)
    growth = np.cumprod(1.0 + values)
    running_max = np.maximum.accumulate(growth)
    drawdown = growth / running_max - 1.0
    std = values.std(ddof=1)
    return {
        "terminal_wealth": float(growth[-1]),
        "return_pct": 100.0 * float(growth[-1] - 1.0),
        "sharpe": float(values.mean() / std * np.sqrt(ANNUALIZATION_SESSIONS)) if std > 0 else 0.0,
        "max_drawdown_pct": 100.0 * float(drawdown.min()),
    }


def evaluate_strategy(replay: dict, *, minimum_positive_years: int) -> dict:
    strategy, benchmark = replay["strategy"], replay["benchmark"]
    metrics = _metrics(strategy)
    reference = _metrics(benchmark)
    annual = pd.DataFrame({"s": strategy, "b": benchmark})
    annual["year"] = pd.to_datetime(annual.index).year
    per_year = annual.groupby("year").apply(
        lambda g: float(np.prod(1 + g["s"]) / np.prod(1 + g["b"])), include_groups=False)
    positive_years = int((per_year > 1.0).sum())
    return {
        "strategy": metrics,
        "benchmark": reference,
        "wealth_ratio_vs_soxx": metrics["terminal_wealth"] / reference["terminal_wealth"],
        "sharpe_difference": metrics["sharpe"] - reference["sharpe"],
        "max_drawdown_improvement_pct": reference["max_drawdown_pct"] - metrics["max_drawdown_pct"],
        "average_exposure": float(replay["exposure"].mean()),
        "total_turnover": float(replay["turnover"].sum()),
        "positive_year_count": positive_years,
        "year_count": int(per_year.size),
        "beats_return": metrics["return_pct"] > reference["return_pct"],
        "beats_sharpe": metrics["sharpe"] > reference["sharpe"],
        "beats_drawdown": metrics["max_drawdown_pct"] > reference["max_drawdown_pct"],
        "stable_excess": bool(metrics["return_pct"] > reference["return_pct"]
                              and metrics["sharpe"] > reference["sharpe"]
                              and metrics["max_drawdown_pct"] > reference["max_drawdown_pct"]
                              and positive_years >= minimum_positive_years),
        "reference": "SOXX buy-and-hold, identical sessions and cost model",
        "scope": SCOPE,
    }
