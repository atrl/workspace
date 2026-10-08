"""Causal risk-budget ablation of a frozen legacy shadow account.

The shadow account supplies prior-close intended weights. Each new strategy
owns its cash, shares and fees; it does not reuse the shadow's executed shares.
The independent ledger is reconciliation evidence only. Qlib owns the native
portfolio report returned by ``run_native_target_replay``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetReplayResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    targets: pd.DataFrame
    config: dict


def _dated(frame: pd.DataFrame, required: set[str]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("blocked: dated DataFrame required")
    result = frame.copy()
    if "date" in result:
        result = result.set_index("date")
    if (not isinstance(result.index, pd.DatetimeIndex) or result.index.tz is not None
            or result.index.has_duplicates or not result.index.is_monotonic_increasing
            or result.empty or not result.index.equals(result.index.normalize())):
        raise ValueError("blocked: unique ordered naive daily dates required")
    if required - set(result):
        raise ValueError(f"blocked: missing columns {sorted(required - set(result))}")
    return result


def build_legacy_risk_targets(legacy, closes: pd.DataFrame, *, volatility_window: int = 60) -> pd.DataFrame:
    """Map each shadow decision to its next-session intended target and cap.

    ``closes`` is Qlib-read SOXL/SOXX close history, with 61 observations needed
    for the registered 60 daily simple-return sample standard deviation.
    The gross intended weight includes the shadow's new signed share request.
    Fees and integer execution are applied separately in each target account.
    The volatility ratio is a historical risk budget, not a risk guarantee.
    """
    if isinstance(volatility_window, bool) or not isinstance(volatility_window, int) or volatility_window < 2:
        raise ValueError("blocked: volatility window must be an integer >= 2")
    prices = _dated(closes, {"SOXL", "SOXX"})
    values = prices[["SOXL", "SOXX"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("blocked: incomplete or nonpositive native closes")
    daily = _dated(legacy.daily, {"close"})
    decisions = legacy.decisions.copy()
    required = {"signal_date", "planned_execution_date", "signal_ready", "should_buy",
                "sell_type", "requested_shares", "signal_account", "signal_exposure"}
    if required - set(decisions):
        raise ValueError("blocked: shadow decisions omit intended-order evidence")
    decisions = decisions.loc[decisions.planned_execution_date.notna()].copy()
    execution = pd.DatetimeIndex(pd.to_datetime(decisions.planned_execution_date))
    signals = pd.DatetimeIndex(pd.to_datetime(decisions.signal_date))
    if not execution.equals(daily.index) or not (signals < execution).all():
        raise ValueError("blocked: complete next-session decision coverage required")
    locations = prices.index.get_indexer(execution)
    if (locations <= 0).any() or not prices.index[locations - 1].equals(signals):
        raise ValueError("blocked: shadow signal must be immediately preceding native session")
    if not decisions.signal_ready.eq(True).all():
        raise ValueError("blocked: shadow signals are not ready")
    if not np.array_equal(daily.close.to_numpy(float), prices.loc[execution, "SOXL"].to_numpy(float)):
        raise ValueError("blocked: shadow/native SOXL closes differ")
    account = decisions.signal_account.to_numpy(float)
    exposure = decisions.signal_exposure.to_numpy(float)
    quantity = decisions.requested_shares.to_numpy(float)
    if (not np.isfinite(np.r_[account, exposure, quantity]).all() or (account <= 0).any()
            or ((exposure < 0) | (exposure > 1)).any() or (quantity < 0).any()
            or not np.equal(quantity, np.trunc(quantity)).all()):
        raise ValueError("blocked: invalid shadow account, exposure or requested shares")
    if not all(isinstance(x, (bool, np.bool_)) for x in decisions.should_buy):
        raise ValueError("blocked: explicit buy decisions required")
    is_buy = decisions.should_buy.to_numpy(bool)
    sells = decisions.sell_type.notna().to_numpy()
    if ((quantity > 0) & ~is_buy & ~sells).any():
        raise ValueError("blocked: nonzero request lacks buy or sell direction")
    signal_close = prices.loc[signals, "SOXL"].to_numpy(float)
    raw = exposure + np.where(is_buy, quantity, -quantity) * signal_close / account
    if ((raw < -1e-12) | (raw > 1 + 1e-12)).any():
        raise ValueError("blocked: intended shadow target is outside funded long-only range")
    # Only round-off at a fully sold/funded boundary is normalized.
    raw = np.clip(raw, 0., 1.)
    sigma = prices[["SOXL", "SOXX"]].pct_change(fill_method=None).rolling(
        volatility_window, min_periods=volatility_window).std(ddof=1).loc[signals]
    if not np.isfinite(sigma.to_numpy()).all() or (sigma.to_numpy() <= 0).any():
        raise ValueError("blocked: complete positive prior-close volatility required")
    cap = np.minimum(1., sigma.SOXX.to_numpy() / sigma.SOXL.to_numpy())
    return pd.DataFrame({"date": execution, "signal_date": signals, "raw_weight": raw,
        "risk_cap": cap, "capped_weight": np.minimum(raw, cap),
        "shadow_exposure": exposure, "signed_requested_shares": np.where(is_buy, quantity, -quantity),
        "signal_close": signal_close, "signal_account": account,
        "sigma_soxl": sigma.SOXL.to_numpy(), "sigma_soxx": sigma.SOXX.to_numpy(),
        "volatility_observations": volatility_window}).set_index("date")


def replay_target_weights(market: pd.DataFrame, targets: pd.DataFrame, *, weight_column: str,
                         cost_bps: float, initial_cash: float = 100_000.) -> TargetReplayResult:
    """Independent funded integer-share accounting at next open, close marks.

    Exact-cost target sizing uses the known previous-close weight and observed
    execution price. Integer rounding leaves less than one share of residual
    target error; this is not a broker fill model or a hard realized risk cap.
    """
    prices = _dated(market, {"open", "close"})
    desired = _dated(targets, {"signal_date", weight_column})
    if not desired.index.isin(prices.index).all():
        raise ValueError("blocked: native market does not cover all target dates")
    if not prices.loc[desired.index[0]:desired.index[-1]].index.equals(desired.index):
        raise ValueError("blocked: target dates must cover every native session in the replay")
    locations = prices.index.get_indexer(desired.index)
    signals = pd.DatetimeIndex(pd.to_datetime(desired.signal_date))
    if (locations <= 0).any() or not prices.index[locations - 1].equals(signals):
        raise ValueError("blocked: target signal must be the previous native session")
    prices = prices.loc[desired.index]
    values = prices[["open", "close"]].to_numpy(float)
    weights = desired[weight_column].to_numpy(float)
    if (not np.isfinite(values).all() or (values <= 0).any()
            or not np.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any()
            or not (signals < desired.index).all()):
        raise ValueError("blocked: invalid target weights, causality or prices")
    if (not np.isfinite([initial_cash, cost_bps]).all() or initial_cash <= 0
            or not 0 <= cost_bps < 10000):
        raise ValueError("blocked: invalid initial cash or proportional cost")
    cash, shares, previous = float(initial_cash), 0, float(initial_cash)
    rate = float(cost_bps) / 10000.
    daily, trades = [], []
    for i, (date, row) in enumerate(prices.iterrows()):
        opening, closing, weight = float(row.open), float(row.close), float(weights[i])
        nav_open = cash + shares * opening
        current_value = shares * opening
        if current_value <= weight * nav_open:
            requested = math.floor(max(0., (weight * nav_open - current_value) /
                                          (opening * (1 + weight * rate))))
            amount = min(requested, math.floor(cash / (opening * (1 + rate))))
            while amount > 0 and amount * opening + amount * opening * rate > cash:
                amount -= 1
        else:
            requested = math.floor(max(0., (current_value - weight * nav_open) /
                                          (opening * (1 - weight * rate))))
            amount = -min(requested, shares)
        notional = abs(amount) * opening
        fee = notional * rate
        cash -= amount * opening + fee
        shares += amount
        if cash < 0 or shares < 0 or not math.isfinite(cash):
            raise AssertionError("blocked: target ledger funding failed")
        if amount:
            trades.append({"date": date, "signal_date": signals[i], "symbol": "SOXL",
                "amount": amount, "price": opening, "fee": fee, "target_weight": weight})
        account = cash + shares * closing
        daily.append({"date": date, "account": account, "cash": cash, "shares": shares,
            "fee": fee, "turnover": notional / previous, "net_return": account / previous - 1.,
            "exposure": shares * closing / account, "open_target_weight": weight,
            "open_exposure_after_cost": shares * opening / (cash + shares * opening)})
        previous = account
    trade_columns = ["date", "signal_date", "symbol", "amount", "price", "fee", "target_weight"]
    return TargetReplayResult(pd.DataFrame(daily), pd.DataFrame(trades, columns=trade_columns), desired,
        {"cost_bps": float(cost_bps), "initial_cash": float(initial_cash), "weight_column": weight_column,
         "execution": "prior-close target, next observed open, integer adjusted-price units",
         "terminal_liquidation": False, "cash_return": 0.})


def run_native_target_replay(market: pd.DataFrame, targets: pd.DataFrame, *, weight_column: str,
                            cost_bps: float, initial_cash: float = 100_000.) -> dict:
    """Require the caller's initialized Qlib provider; native report is primary."""
    from src.workflow.legacy_soxl_repair import native_order_backtest, reconcile_engine
    engine = replay_target_weights(market, targets, weight_column=weight_column,
        cost_bps=cost_bps, initial_cash=initial_cash)
    native = native_order_backtest(dates=engine.daily.date, cost_bps=cost_bps,
        initial_cash=initial_cash, trades=engine.trades)
    audit = reconcile_engine(engine.daily, native)
    return {"native": native, "engine": engine, "daily": engine.daily,
            "trades": engine.trades, "audit": audit, "reconciliation": audit}
