"""Causal, non-leveraged exposure policies measured against the SOXX benchmark.

This module answers one narrow question with no fitted return signal: given only
past-available underlying risk and trend observations, can a long-only
SOXX/SOXL/cash policy deliver *stable* excess over buy-and-hold SOXX on both
return and drawdown at the same time?

Every exposure is computed from observations through the signal close and is
applied to the next session's return. There is no look-ahead, no return-mean
prediction, no fitted alpha, no parameter search inside this module. The caller
supplies a bounded, pre-registered candidate family and owns segmentation and
selection. This analyzer does not grant production eligibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

ASSETS = ("SOXL", "SOXX", "cash")
ANNUALIZATION_SESSIONS = 252
SCOPE = "reused_history_and_forward_diagnostic_not_production_evidence"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _positive_int(value, name):
    _require(isinstance(value, Integral) and not isinstance(value, bool) and value > 1,
             name + " must be an integer > 1")
    return int(value)


def _nonnegative_real(value, name):
    _require(isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value) and value >= 0,
             name + " must be a finite nonnegative real")
    return float(value)


@dataclass(frozen=True)
class ExposurePolicy:
    """A pre-registered causal exposure rule.

    ``kind`` selects the mapping from past risk/trend state to target weights:

    - ``soxx_buy_hold``: always fully invested in SOXX. This is the benchmark.
    - ``vol_target_soxx``: SOXX exposure ``min(1, target_vol / realized_vol)``,
      the remainder in cash. Never holds SOXL.
    - ``trend_vol_switch``: hold SOXL only when SOXX is above its long moving
      average *and* realized volatility is below ``vol_cap``; otherwise hold SOXX
      when above the moving average, else cash.
    """

    name: str
    kind: str
    vol_window: int = 20
    trend_window: int = 200
    target_vol: float = 0.20
    vol_cap: float = 0.20

    def validate(self):
        _require(isinstance(self.name, str) and self.name, "policy name is required")
        _require(self.kind in ("soxx_buy_hold", "vol_target_soxx", "trend_vol_switch"),
                 "unknown exposure policy kind")
        _positive_int(self.vol_window, "vol_window")
        _positive_int(self.trend_window, "trend_window")
        _nonnegative_real(self.target_vol, "target_vol")
        _nonnegative_real(self.vol_cap, "vol_cap")
        return self


def _price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    _require(isinstance(prices, pd.DataFrame) and not prices.empty, "prices frame is required")
    missing = {"signal_date", "soxl_close", "soxx_close"} - set(prices.columns)
    _require(not missing, f"prices frame missing columns: {sorted(missing)}")
    frame = prices.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    _require(not frame["signal_date"].duplicated().any(), "signal dates must be unique")
    frame = frame.sort_values("signal_date").reset_index(drop=True)
    _require(np.isfinite(frame[["soxl_close", "soxx_close"]].to_numpy(float)).all(),
             "closes must be finite")
    _require((frame[["soxl_close", "soxx_close"]] > 0).to_numpy().all(), "closes must be positive")
    return frame


def build_state(prices: pd.DataFrame, *, vol_window: int = 20, trend_window: int = 200) -> pd.DataFrame:
    """Attach strictly past-available realized volatility and trend state.

    Realized volatility uses simple returns through the signal close. The trend
    state uses the SOXX close versus its trailing moving average. All state at a
    row is computable before the next session opens.
    """
    vol_window = _positive_int(vol_window, "vol_window")
    trend_window = _positive_int(trend_window, "trend_window")
    frame = _price_frame(prices)
    soxx_ret = frame["soxx_close"].pct_change()
    frame["soxx_return_realized"] = soxx_ret
    frame["soxl_return_realized"] = frame["soxl_close"].pct_change()
    frame["realized_vol"] = soxx_ret.rolling(vol_window).std(ddof=1) * np.sqrt(ANNUALIZATION_SESSIONS)
    frame["trend_ma"] = frame["soxx_close"].rolling(trend_window).mean()
    frame["above_trend"] = frame["soxx_close"] > frame["trend_ma"]
    return frame


def target_weights(state: pd.Series, policy: ExposurePolicy) -> np.ndarray:
    """Map one past-state row to a funded three-asset target (SOXL, SOXX, cash)."""
    policy.validate()
    soxx_ret = float(state["soxx_return_realized"])
    soxl_ret = float(state["soxl_return_realized"])
    realized_vol = float(state["realized_vol"])
    above_trend = bool(state["above_trend"])
    _require(np.isfinite([soxx_ret, soxl_ret, realized_vol]).all(), "state must be finite")
    if policy.kind == "soxx_buy_hold":
        return np.array([0.0, 1.0, 0.0])
    if policy.kind == "vol_target_soxx":
        _require(realized_vol > 0, "realized volatility must be positive for vol targeting")
        exposure = min(1.0, policy.target_vol / realized_vol)
        _require(0.0 <= exposure <= 1.0, "vol target exposure out of range")
        return np.array([0.0, exposure, 1.0 - exposure])
    # trend_vol_switch
    low_vol = realized_vol > 0 and realized_vol < policy.vol_cap
    if above_trend and low_vol:
        return np.array([1.0, 0.0, 0.0])
    if above_trend:
        return np.array([0.0, 1.0, 0.0])
    return np.array([0.0, 0.0, 1.0])


def simulate_policy(state: pd.DataFrame, policy: ExposurePolicy, *, cost_bps: float) -> pd.DataFrame:
    """Replay a causal policy with next-session returns and owned turnover fees.

    Exposure computed from row ``t`` state is applied to the ``t+1`` realized
    return. Turnover from ``t-1`` target to ``t`` target is charged at ``t``.
    """
    policy.validate()
    _require(isinstance(cost_bps, Real) and not isinstance(cost_bps, bool) and np.isfinite(cost_bps)
             and cost_bps >= 0, "cost_bps must be a finite nonnegative real")
    _require(isinstance(state, pd.DataFrame) and not state.empty, "state frame is required")
    required = {"signal_date", "realized_vol", "above_trend", "soxx_return_realized", "soxl_return_realized"}
    _require(not (required - set(state.columns)), "state is missing risk/trend columns")
    state = state.reset_index(drop=True)
    targets = np.asarray([target_weights(row, policy) for _, row in state.iterrows()], dtype="float64")
    # next-session realized asset returns
    next_soxl = state["soxl_return_realized"].shift(-1)
    next_soxx = state["soxx_return_realized"].shift(-1)
    records = []
    prior_target = np.array([0.0, 0.0, 1.0])
    fee_rate = float(cost_bps) / 10_000.0
    for i in range(len(state)):
        target = targets[i]
        turnover = float(np.abs(target - prior_target).sum())
        soxl_r = next_soxl.iloc[i]
        soxx_r = next_soxx.iloc[i]
        if pd.isna(soxl_r) or pd.isna(soxx_r):
            break
        gross = target[0] * float(soxl_r) + target[1] * float(soxx_r)
        fee = turnover * fee_rate
        records.append({
            "signal_date": state["signal_date"].iloc[i],
            "soxl_weight": target[0],
            "soxx_weight": target[1],
            "cash_weight": target[2],
            "turnover": turnover,
            "gross_return": gross,
            "strategy_return": gross - fee,
            "soxl_return": float(soxl_r),
            "soxx_return": float(soxx_r),
        })
        prior_target = target
    result = pd.DataFrame(records)
    _require(not result.empty, "policy simulation produced no sessions")
    return result


def risk_metrics(returns: np.ndarray) -> dict:
    """Absolute and risk-adjusted statistics for a daily return path."""
    values = np.asarray(returns, dtype="float64")
    _require(values.size > 0 and np.isfinite(values).all(), "returns must be finite and nonempty")
    _require((values > -1.0).all(), "returns must exceed -100%")
    growth = np.cumprod(1.0 + values)
    terminal = float(growth[-1])
    sessions = values.size
    years = sessions / ANNUALIZATION_SESSIONS
    cagr = terminal ** (1.0 / years) - 1.0 if terminal > 0 else -1.0
    running_max = np.maximum.accumulate(growth)
    drawdown = growth / running_max - 1.0
    volatility = float(values.std(ddof=1) * np.sqrt(ANNUALIZATION_SESSIONS)) if sessions > 1 else 0.0
    sharpe = float(values.mean() / values.std(ddof=1) * np.sqrt(ANNUALIZATION_SESSIONS)) if values.std(ddof=1) > 0 else 0.0
    tail = np.sort(values)[: max(1, int(np.ceil(0.05 * sessions)))]
    return {
        "sessions": int(sessions),
        "terminal_wealth": terminal,
        "return_pct": 100.0 * (terminal - 1.0),
        "cagr_pct": 100.0 * cagr,
        "annualized_volatility_pct": 100.0 * volatility,
        "sharpe": sharpe,
        "max_drawdown_pct": 100.0 * float(drawdown.min()),
        "worst_day_pct": 100.0 * float(values.min()),
        "cvar5_pct": 100.0 * float(tail.mean()),
    }


def excess_versus_benchmark(policy_replay: pd.DataFrame, benchmark_replay: pd.DataFrame) -> dict:
    """Compare a policy path with SOXX buy-and-hold on aligned sessions."""
    _require(len(policy_replay) == len(benchmark_replay), "policy and benchmark sessions must align")
    _require((policy_replay["signal_date"].to_numpy() == benchmark_replay["signal_date"].to_numpy()).all(),
             "policy and benchmark dates must align")
    strategy = policy_replay["strategy_return"].to_numpy(float)
    benchmark = benchmark_replay["strategy_return"].to_numpy(float)
    strategy_metrics = risk_metrics(strategy)
    benchmark_metrics = risk_metrics(benchmark)
    log_excess = np.log1p(strategy) - np.log1p(benchmark)
    return {
        "strategy": strategy_metrics,
        "benchmark": benchmark_metrics,
        "terminal_wealth_ratio_vs_soxx": strategy_metrics["terminal_wealth"] / benchmark_metrics["terminal_wealth"],
        "cumulative_log_excess": float(log_excess.sum()),
        "annualized_log_excess_pct": 100.0 * float(log_excess.mean() * ANNUALIZATION_SESSIONS),
        "sharpe_difference": strategy_metrics["sharpe"] - benchmark_metrics["sharpe"],
        "max_drawdown_improvement_pct": benchmark_metrics["max_drawdown_pct"] - strategy_metrics["max_drawdown_pct"],
        "cvar5_improvement_pct": benchmark_metrics["cvar5_pct"] - strategy_metrics["cvar5_pct"],
        "beats_return": strategy_metrics["return_pct"] > benchmark_metrics["return_pct"],
        "beats_drawdown": strategy_metrics["max_drawdown_pct"] > benchmark_metrics["max_drawdown_pct"],
        "beats_sharpe": strategy_metrics["sharpe"] > benchmark_metrics["sharpe"],
        "average_soxl_weight": float(policy_replay["soxl_weight"].mean()),
        "average_cash_weight": float(policy_replay["cash_weight"].mean()),
        "total_turnover": float(policy_replay["turnover"].sum()),
    }


def forecast_information_coefficient(forecast: np.ndarray, realized: np.ndarray) -> dict:
    """Descriptive IC of a frozen forecast against realized returns.

    This is a description, never a transformation of the forecast. It exists so
    a card can state, with one number, whether a return signal carries stable
    information before any allocation is built on top of it.
    """
    prediction = np.asarray(forecast, dtype="float64")
    outcome = np.asarray(realized, dtype="float64")
    _require(prediction.shape == outcome.shape and prediction.size > 2,
             "forecast and realized arrays must be aligned and non-trivial")
    _require(np.isfinite(prediction).all() and np.isfinite(outcome).all(), "inputs must be finite")
    if np.ptp(prediction) == 0 or np.ptp(outcome) == 0:
        return {"pearson_ic": None, "spearman_ic": None, "degenerate": True}
    rank_corr = pd.Series(prediction).corr(pd.Series(outcome), method="spearman")
    return {
        "pearson_ic": float(np.corrcoef(prediction, outcome)[0, 1]),
        "spearman_ic": float(rank_corr),
        "degenerate": False,
        "scope": "descriptive_only_no_forecast_transformation",
    }


def intraday_early_predictive_correlation(early_return: np.ndarray, later_return: np.ndarray) -> dict:
    """Correlation of the opening path with the remainder of the session.

    Used to test whether an observable early-session move predicts the close.
    A near-zero correlation means any intraday edge must come from risk/state
    conditioning, not from return prediction.
    """
    early = np.asarray(early_return, dtype="float64")
    later = np.asarray(later_return, dtype="float64")
    _require(early.shape == later.shape and early.size > 2, "arrays must be aligned and non-trivial")
    mask = np.isfinite(early) & np.isfinite(later)
    early, later = early[mask], later[mask]
    _require(early.size > 2, "too few finite observations")
    if np.ptp(early) == 0 or np.ptp(later) == 0:
        return {"pearson_correlation": None, "degenerate": True}
    positive = early > 0
    return {
        "pearson_correlation": float(np.corrcoef(early, later)[0, 1]),
        "later_mean_when_early_positive_pct": 100.0 * float(later[positive].mean()),
        "later_mean_when_early_negative_pct": 100.0 * float(later[~positive].mean()),
        "observations": int(early.size),
        "degenerate": False,
    }


def annual_excess(policy_replay: pd.DataFrame, benchmark_replay: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-year return comparison, aligned on the same sessions."""
    policy = policy_replay.copy()
    benchmark = benchmark_replay.copy()
    policy["year"] = pd.to_datetime(policy["signal_date"]).dt.year
    benchmark["year"] = pd.to_datetime(benchmark["signal_date"]).dt.year
    rows = []
    for year, group in policy.groupby("year"):
        bench = benchmark[benchmark["year"].eq(year)]
        _require(len(group) == len(bench), "annual policy/benchmark sessions must align")
        strategy_wealth = float(np.prod(1.0 + group["strategy_return"].to_numpy(float)))
        benchmark_wealth = float(np.prod(1.0 + bench["strategy_return"].to_numpy(float)))
        rows.append({
            "year": int(year),
            "sessions": int(len(group)),
            "strategy_return_pct": 100.0 * (strategy_wealth - 1.0),
            "benchmark_return_pct": 100.0 * (benchmark_wealth - 1.0),
            "wealth_ratio_vs_soxx": strategy_wealth / benchmark_wealth,
            "average_soxl_weight": float(group["soxl_weight"].mean()),
            "average_cash_weight": float(group["cash_weight"].mean()),
        })
    return pd.DataFrame(rows)


def stable_excess_verdict(excess: dict, annual: pd.DataFrame, *, minimum_positive_years: int) -> dict:
    """Two-sided stable-excess test: return AND drawdown must beat SOXX."""
    _require(isinstance(minimum_positive_years, Integral) and minimum_positive_years >= 1,
             "minimum_positive_years must be a positive integer")
    positive_years = int(annual["wealth_ratio_vs_soxx"].gt(1.0).sum())
    years = int(len(annual))
    return {
        "beats_return_and_drawdown": bool(excess["beats_return"] and excess["beats_drawdown"]),
        "beats_sharpe": bool(excess["beats_sharpe"]),
        "positive_year_count": positive_years,
        "year_count": years,
        "minimum_positive_years": int(minimum_positive_years),
        "annual_win_rate": positive_years / years if years else 0.0,
        "stable_excess": bool(
            excess["beats_return"]
            and excess["beats_drawdown"]
            and excess["beats_sharpe"]
            and positive_years >= minimum_positive_years
        ),
        "reference": "SOXX buy-and-hold, identical sessions and cost model",
        "scope": SCOPE,
    }
