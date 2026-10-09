"""Night-session (overnight futures / premarket) signals for the US regular session.

The collected data has two genuinely overnight information sources that are
observable before the 09:30 US cash open:

- Equity-index futures (ES/NQ) trade a near-continuous session; the change from
  the prior 18:00 ET to the current 09:30 ET is known at the cash open.
- Premarket equity bars (04:00-09:30 ET) for SOXL/SOXX/SMH.

A signal is only causal if it is computed from bars that end at or before the
cash open. The traded target is the same day's regular session (09:30-16:00 ET)
return. This module never uses a regular-session bar to build a same-day signal.

No return forecast, fitted parameter or future information is used.
"""
from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

ANNUALIZATION_SESSIONS = 252
SCOPE = "new_overnight_data_diagnostic_not_production_evidence"
PREMARKET = ("04:00", "09:30")
REGULAR = ("09:30", "16:00")
AFTERHOURS = ("16:00", "20:00")
OVERNIGHT_START = "18:00"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def load_night_session_instruments(data_dir) -> dict[str, pd.DataFrame]:
    """Load the raw night-session parquet files keyed by lowercase instrument."""
    from pathlib import Path

    base = Path(data_dir)
    _require(base.is_dir(), f"night-session data directory is missing: {base}")
    files = {"soxl": "SOXL", "soxx": "SOXX", "smh": "SMH", "es": "ES_F",
             "nq": "NQ_F", "nvda": "NVDA", "amd": "AMD", "tsm": "TSM"}
    instruments = {}
    for key, name in files.items():
        path = base / f"{name}.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index)
        instruments[key] = frame
    _require("soxl" in instruments and "soxx" in instruments, "SOXL/SOXX night-session data is required")
    return instruments


def _window(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_time = pd.Timestamp(start).time()
    end_time = pd.Timestamp(end).time()
    mask = (frame.index.time >= start_time) & (frame.index.time < end_time)
    return frame.loc[mask]


def regular_session(frame: pd.DataFrame) -> pd.DataFrame:
    """Open and close of the 09:30-16:00 regular session per day."""
    block = _window(frame, *REGULAR).copy()
    block["day"] = block.index.normalize()
    grouped = block.groupby("day")
    return pd.DataFrame({"open": grouped["open"].first(), "close": grouped["close"].last()})


def premarket_return(frame: pd.DataFrame) -> pd.Series:
    """Premarket return 04:00 to 09:30, known before the cash open."""
    block = _window(frame, *PREMARKET)
    block = block.assign(day=block.index.normalize())
    grouped = block.groupby("day")
    return (grouped["close"].last() / grouped["open"].first() - 1.0).rename("premarket_return")


def futures_overnight_return(frame: pd.DataFrame) -> pd.Series:
    """Futures change from prior 18:00 ET to current 09:30 ET, per US session day.

    This is the true overnight window: it starts after the prior cash close and
    ends at the current cash open, so it is fully observable before trading.
    """
    _require(isinstance(frame.index, pd.DatetimeIndex), "frame needs a DatetimeIndex")
    work = frame.assign(day=frame.index.normalize()).sort_index()
    days = sorted(work["day"].unique())
    rows = {}
    start_time = pd.Timestamp(OVERNIGHT_START).time()
    open_time = pd.Timestamp(REGULAR[0]).time()
    for index, day in enumerate(days):
        if index == 0:
            continue
        previous = days[index - 1]
        prior_evening = (work["day"] == previous) & (work.index.time >= start_time)
        current_morning = (work["day"] == day) & (work.index.time < open_time)
        window = work.loc[prior_evening | current_morning]
        if len(window) >= 2:
            rows[day] = window["close"].iloc[-1] / window["open"].iloc[0] - 1.0
    return pd.Series(rows, name="overnight_return").sort_index()


def build_session_panel(instruments: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble a per-day panel of causal overnight signals and session returns.

    ``instruments`` must contain ``soxl`` and ``soxx`` regular-session frames plus
    any of ``es``/``nq`` futures or ``smh``/``nvda``/``tsm`` premarket frames.
    """
    _require("soxl" in instruments and "soxx" in instruments, "soxl and soxx are required")
    soxl = regular_session(instruments["soxl"]).rename(columns={"open": "soxl_open", "close": "soxl_close"})
    soxx = regular_session(instruments["soxx"]).rename(columns={"open": "soxx_open", "close": "soxx_close"})
    panel = soxl.join(soxx, how="inner")
    panel["soxl_return"] = panel["soxl_close"] / panel["soxl_open"] - 1.0
    panel["soxx_return"] = panel["soxx_close"] / panel["soxx_open"] - 1.0
    for key in ("es", "nq"):
        if key in instruments:
            panel[f"{key}_overnight"] = futures_overnight_return(instruments[key])
    for key in ("soxl", "soxx", "smh", "nvda", "tsm"):
        if key in instruments:
            panel[f"{key}_premarket"] = premarket_return(instruments[key])
    panel = panel[panel.index.dayofweek < 5]
    return panel


def switch_strategy(signal: pd.Series, soxl_return: pd.Series, soxx_return: pd.Series, *,
                    threshold: float, cost_bps: float, defensive: str = "soxx") -> pd.DataFrame:
    """Hold SOXL when the overnight signal exceeds ``threshold``, else the defensive asset.

    ``defensive`` selects the fallback when the signal is not positive: ``soxx``
    (relative switch) or ``cash`` (absolute switch, zero return).
    """
    _require(cost_bps >= 0, "cost_bps must be nonnegative")
    _require(defensive in ("soxx", "cash"), "defensive must be soxx or cash")
    soxl_weight = (signal > threshold).astype(float)
    defensive_weight = 1.0 - soxl_weight
    soxx_weight = defensive_weight if defensive == "soxx" else soxl_weight * 0.0
    cash_weight = defensive_weight if defensive == "cash" else soxl_weight * 0.0
    frame = pd.DataFrame({"soxl_w": soxl_weight, "soxx_w": soxx_weight, "cash_w": cash_weight,
                          "soxl": soxl_return, "soxx": soxx_return}).dropna()
    _require(len(frame) > 20, "not enough sessions")
    turnover = ((frame["soxl_w"] - frame["soxl_w"].shift(1).fillna(0.0)).abs()
                + (frame["soxx_w"] - frame["soxx_w"].shift(1).fillna(0.0)).abs()
                + (frame["cash_w"] - frame["cash_w"].shift(1).fillna(0.0)).abs())
    strategy = (frame["soxl_w"] * frame["soxl"] + frame["soxx_w"] * frame["soxx"]
                - turnover * float(cost_bps) / 1e4)
    return pd.DataFrame({"strategy": strategy, "benchmark": frame["soxx"],
                         "exposure": frame["soxl_w"], "turnover": turnover})


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


def evaluate_strategy(replay: pd.DataFrame, *, minimum_positive_years: int) -> dict:
    strategy, benchmark = replay["strategy"], replay["benchmark"]
    metrics, reference = _metrics(strategy), _metrics(benchmark)
    annual = pd.DataFrame({"s": strategy, "b": benchmark})
    annual["year"] = pd.to_datetime(annual.index).year
    per_year = annual.groupby("year").apply(
        lambda g: float(np.prod(1 + g["s"]) / np.prod(1 + g["b"])), include_groups=False)
    positive_years = int((per_year > 1.0).sum())
    return {
        "strategy": metrics, "benchmark": reference,
        "wealth_ratio_vs_soxx": metrics["terminal_wealth"] / reference["terminal_wealth"],
        "sharpe_difference": metrics["sharpe"] - reference["sharpe"],
        "max_drawdown_improvement_pct": reference["max_drawdown_pct"] - metrics["max_drawdown_pct"],
        "average_exposure": float(replay["exposure"].mean()),
        "total_turnover": float(replay["turnover"].sum()),
        "switches_per_year": float(replay["turnover"].sum() / (len(replay) / ANNUALIZATION_SESSIONS)),
        "positive_year_count": positive_years,
        "year_count": int(per_year.size),
        "beats_return": metrics["return_pct"] > reference["return_pct"],
        "beats_sharpe": metrics["sharpe"] > reference["sharpe"],
        "beats_drawdown": metrics["max_drawdown_pct"] > reference["max_drawdown_pct"],
        "stable_excess": bool(metrics["return_pct"] > reference["return_pct"]
                              and metrics["sharpe"] > reference["sharpe"]
                              and metrics["max_drawdown_pct"] > reference["max_drawdown_pct"]
                              and positive_years >= minimum_positive_years),
        "reference": "SOXX buy-and-hold, identical regular-session window and cost model",
        "scope": SCOPE,
    }


def information_coefficient(signal: pd.Series, soxl_return: pd.Series,
                            soxx_return: pd.Series) -> dict:
    frame = pd.DataFrame({"x": signal, "soxl": soxl_return, "soxx": soxx_return}).dropna()
    _require(len(frame) > 60, "not enough observations")
    return {
        "ic_soxl": float(frame["x"].corr(frame["soxl"])),
        "ic_excess": float(frame["x"].corr(frame["soxl"] - frame["soxx"])),
        "observations": int(len(frame)),
    }
