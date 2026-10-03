"""Causal source audits and native Qlib reconciliation for regime research.

The open reconciliation executes a prescribed open-only schedule through Qlib.
It validates settlement and provider alignment; it is not an independent replay
of a controller, an intraday recovery backtest, or evidence of live fills.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SYMBOLS = ("SOXL", "SOXX")
FIRST_HOUR_CLOSES = ("09:45", "10:00", "10:15", "10:30")
BAR_FIELDS = ("$open", "$high", "$low", "$close", "$volume", "$bar_count")
EARLY_FIELDS = ("early_ret", "early_range", "early_from_low", "early_volume", "early_bar_count")


def _read_first_hour_bars(provider_uri: Path, *, start: str, end: str) -> pd.DataFrame:
    import qlib
    from qlib.data import D

    if not provider_uri.is_dir():
        raise ValueError("blocked: first-hour Qlib provider is unavailable")
    qlib.init(provider_uri=str(provider_uri), region="us", kernels=1)
    # Include the full session to detect the old head(4) loader's possible use
    # of late observations when one of the actual first four bars is missing.
    return D.features(list(SYMBOLS), list(BAR_FIELDS), start_time=start,
                      end_time=f"{end} 16:00:00", freq="15min").reset_index()


def audit_first_hour_inputs(frame: pd.DataFrame, *, provider_uri: str | Path):
    """Return a copy with invalid recovery inputs disabled, a day audit, summary.

    Exact close-stamped 15-minute bars are required. ``bar_count`` in this
    provider is the total number of session bars repeated on every row; it is
    never interpreted as observed first-hour minute coverage. The original
    frame and its immutable recorder artifact are not modified.
    """
    required = {"signal_date", "trade_date", *[f"{name}_{s.lower()}" for name in EARLY_FIELDS for s in SYMBOLS]}
    if required - set(frame):
        raise ValueError(f"blocked: incomplete retained first-hour inputs: {sorted(required - set(frame))}")
    result = frame.copy(deep=True)
    dates = pd.to_datetime(result.trade_date)
    if result.empty or dates.isna().any() or dates.duplicated().any():
        raise ValueError("blocked: first-hour audit requires unique finite trade dates")
    provider = Path(provider_uri).resolve()
    manifest_path = provider / "qlib_store_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("blocked: first-hour provider manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    bars = _read_first_hour_bars(provider, start=str(dates.min().date()), end=str(dates.max().date()))
    bars["datetime"] = pd.to_datetime(bars.datetime)
    if bars.datetime.dt.tz is not None:
        bars["datetime"] = bars.datetime.dt.tz_convert("America/New_York").dt.tz_localize(None)
    bars["date"] = bars.datetime.dt.normalize()
    bars = bars.sort_values(["instrument", "date", "datetime"])
    groups = {key: group.sort_values("datetime") for key, group in bars.groupby(["instrument", "date"])}
    first_four = bars.groupby(["instrument", "date"], group_keys=False).head(4)
    # Preserve the provider's float32 aggregation semantics. Converting each
    # price to float64 first changes tiny ratios and large-volume summation.
    recomputed = first_four.groupby(["instrument", "date"]).agg(
        early_open=("$open", "first"), early_close=("$close", "last"),
        early_high=("$high", "max"), early_low=("$low", "min"),
        early_volume=("$volume", "sum"), early_bar_count=("$bar_count", "sum"),
    )
    recomputed["early_ret"] = recomputed.early_close / recomputed.early_open - 1.0
    recomputed["early_range"] = recomputed.early_low / recomputed.early_high - 1.0
    recomputed["early_from_low"] = recomputed.early_close / recomputed.early_low - 1.0
    all_early = [name for name in result if name.startswith("early_")]
    audit_rows = []
    for index, row in result.iterrows():
        day = pd.Timestamp(row.trade_date).normalize()
        details, valid, has_retained = [], True, False
        maximum_error = 0.0
        for symbol in SYMBOLS:
            suffix = symbol.lower()
            columns = [f"{name}_{suffix}" for name in EARLY_FIELDS]
            observed = pd.to_numeric(row[columns], errors="coerce").to_numpy(dtype=float)
            has_retained |= bool(np.isfinite(observed).any())
            group = groups.get((symbol, day))
            if group is None or group.empty:
                details.append(f"{symbol}:provider_unavailable")
                valid = False
                continue
            early = group.head(4)
            timestamps = tuple(early.datetime.dt.strftime("%H:%M"))
            finite = np.isfinite(early.loc[:, list(BAR_FIELDS)].to_numpy(dtype=float)).all()
            if timestamps != FIRST_HOUR_CLOSES or not finite:
                details.append(f"{symbol}:invalid_first_hour_bars:{','.join(timestamps)}")
                valid = False
                continue
            prices = early.loc[:, ["$open", "$high", "$low", "$close"]].to_numpy(dtype=float)
            if (prices <= 0).any() or (early["$volume"] < 0).any():
                details.append(f"{symbol}:invalid_first_hour_prices_or_volume")
                valid = False
                continue
            computed = recomputed.loc[(symbol, day), list(EARLY_FIELDS)].to_numpy(dtype=float)
            if not np.isfinite(observed).all() or not np.allclose(observed, computed, rtol=1e-10, atol=1e-10):
                details.append(f"{symbol}:retained_feature_mismatch_or_missing")
                valid = False
            if np.isfinite(observed).all():
                maximum_error = max(maximum_error, float(np.abs(observed - computed).max()))
        if not valid:
            result.loc[index, all_early] = np.nan
        local_close = day.tz_localize("America/New_York") + pd.Timedelta(hours=10, minutes=30)
        audit_rows.append({
            "signal_date": row.signal_date, "trade_date": day,
            "status": "valid" if valid else ("invalid_retained_observation" if has_retained else "not_available"),
            "valid_for_recovery": valid, "retained_observation_present": has_retained,
            "feature_available_at": local_close.tz_convert("UTC").isoformat() if valid else None,
            "recovery_decision_at": (local_close + pd.Timedelta(minutes=1)).tz_convert("UTC").isoformat(),
            "expected_close_stamps": ",".join(FIRST_HOUR_CLOSES),
            "maximum_retained_feature_error": maximum_error,
            "reasons": ";".join(details),
        })
    audit = pd.DataFrame(audit_rows)
    summary = {
        "status": "passed" if not audit.status.eq("invalid_retained_observation").any() else "invalid_recovery_observations_disabled",
        "provider_uri": str(provider), "provider_as_of": manifest.get("as_of"),
        "provider_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_api": ["qlib.init", "qlib.data.D.features"], "rows": len(audit),
        "valid_recovery_rows": int(audit.valid_for_recovery.sum()),
        "unavailable_recovery_rows": int(audit.status.eq("not_available").sum()),
        "invalid_retained_recovery_rows": int(audit.status.eq("invalid_retained_observation").sum()),
        "maximum_retained_feature_error": float(audit.maximum_retained_feature_error.max()),
        "source_unchanged": True,
        "bar_count_semantics": "full-session 15-minute bar count, not observed first-hour minute coverage; not a model feature",
        "limitation": "Exact aggregate bar timestamps do not prove every underlying one-minute observation is present.",
    }
    return result, audit, summary


def native_open_reconciliation(frame: pd.DataFrame, replay: pd.DataFrame, *, provider_uri: str | Path,
                               cost_bps: float = 10.0, account: float = 100_000.0) -> dict[str, Any]:
    """Execute and reconcile a contiguous, initially-cash, open-only schedule.

    Qlib's report marks at the same-day close. Separate next-open marks of its
    actual Position objects reconcile the research session accounting without
    relabeling daily close reports as open-to-open returns.
    """
    import qlib
    from qlib.backtest import backtest
    from qlib.backtest.decision import Order, TradeDecisionWO
    from qlib.contrib.evaluate import risk_analysis
    from qlib.data import D
    from src.backtest.strategy import DirectTargetWeightStrategy
    from src.workflow.stage1_economic_action import fee_affordable_target

    if replay.empty or len(frame) != len(replay) or not np.isfinite([cost_bps, account]).all() or cost_bps < 0 or account <= 0:
        raise ValueError("blocked: invalid Qlib reconciliation inputs")
    if replay.recovery_action.astype(bool).any():
        raise ValueError("blocked: daily native reconciliation cannot represent intraday recovery")
    market = frame.copy().reset_index(drop=True)
    schedule = replay.copy().reset_index(drop=True)
    for column in ("signal_date", "trade_date", "exit_date"):
        market[column], schedule[column] = pd.to_datetime(market[column]), pd.to_datetime(schedule[column])
        if not market[column].equals(schedule[column]):
            raise ValueError("blocked: native reconciliation schedule dates differ")
    if (not schedule.trade_date.is_monotonic_increasing or schedule.trade_date.duplicated().any()
            or not schedule.signal_date.lt(schedule.trade_date).all()
            or not schedule.trade_date.lt(schedule.exit_date).all()
            or not schedule.exit_date.iloc[:-1].reset_index(drop=True).equals(schedule.trade_date.iloc[1:].reset_index(drop=True))):
        raise ValueError("blocked: native reconciliation requires contiguous causal sessions")
    provider = Path(provider_uri).resolve()
    if not provider.is_dir():
        raise ValueError("blocked: native reconciliation Qlib provider is unavailable")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    first, last, exit_day = schedule.trade_date.iloc[0], schedule.trade_date.iloc[-1], schedule.exit_date.iloc[-1]
    prices = D.features(list(SYMBOLS), ["$open", "$close"], start_time=first, end_time=exit_day, freq="day")
    opens = prices["$open"].unstack("instrument").astype(float)
    if not np.isfinite(opens.reindex(pd.DatetimeIndex([*schedule.trade_date, exit_day])).to_numpy()).all():
        raise ValueError("blocked: missing native open prices")
    for symbol in SYMBOLS:
        returns = opens[symbol].reindex(schedule.exit_date).to_numpy() / opens[symbol].reindex(schedule.trade_date).to_numpy() - 1
        if not np.allclose(returns, market[f"{symbol.lower()}_return"].to_numpy(), rtol=0, atol=1e-7):
            raise ValueError("blocked: retained returns disagree with Qlib open prices")

    class NativeOpenTargetSchedule(DirectTargetWeightStrategy):
        def __init__(self):
            super().__init__(signal=pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])), risk_degree=1.0)
            self.schedule = schedule.set_index("trade_date")

        def generate_trade_decision(self, execute_result=None):
            step = self.trade_calendar.get_trade_step()
            start, end = self.trade_calendar.get_step_time(step)
            row = self.schedule.loc[pd.Timestamp(start).normalize()]
            if not bool(row.execute_open_target):
                return TradeDecisionWO([], self)
            current = self.trade_position.get_stock_amount_dict()
            quote = {symbol: self.trade_exchange.get_deal_price(symbol, start, end, direction=Order.BUY) for symbol in SYMBOLS}
            values = np.array([current.get(symbol, 0.0) * quote[symbol] for symbol in SYMBOLS] + [self.trade_position.get_cash()])
            nav = values.sum()
            desired = row[["open_soxl_weight", "open_soxx_weight", "open_cash_weight"]].to_numpy(dtype=float)
            executable = fee_affordable_target(desired, pretrade_weights=values / nav, cost_bps=cost_bps)
            target = {symbol: nav * executable[i] / quote[symbol] for i, symbol in enumerate(SYMBOLS)}
            orders = self.trade_exchange.generate_order_for_target_amount_position(target, current, start, end)
            return TradeDecisionWO(orders, self)

    native, indicators = backtest(
        start_time=first, end_time=last, strategy=NativeOpenTargetSchedule(),
        executor={"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                  "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}},
        account=account, benchmark="SOXX",
        exchange_kwargs={"codes": list(SYMBOLS), "freq": "day", "deal_price": "open", "limit_threshold": None,
                         "trade_unit": None, "open_cost": cost_bps / 10_000., "close_cost": cost_bps / 10_000., "min_cost": 0.0},
    )
    report, positions = native["1day"]
    reconciled = []
    previous_nav = float(account)
    for row in schedule.itertuples():
        position = positions[row.trade_date]
        cash = float(position.get_cash())
        quantities = position.get_stock_amount_dict()
        exit_nav = cash + sum(quantities.get(symbol, 0.0) * opens.loc[row.exit_date, symbol] for symbol in SYMBOLS)
        native_return = exit_nav / previous_nav - 1.0
        reconciled.append({"trade_date": row.trade_date, "exit_date": row.exit_date,
                           "native_next_open_nav": exit_nav, "native_next_open_return": native_return,
                           "research_return": float(row.strategy_return), "return_error": native_return - row.strategy_return})
        previous_nav = exit_nav
    reconciliation = pd.DataFrame(reconciled)
    maximum_error = float(reconciliation.return_error.abs().max())
    close_net = report["return"] - report["cost"]
    summary = {"status": "passed" if maximum_error <= 1e-7 else "failed",
               "sessions": len(schedule), "start": str(first.date()), "end": str(last.date()),
               "provider_uri": str(provider), "cost_bps": float(cost_bps), "initial_cash": float(account),
               "maximum_session_return_error": maximum_error, "tolerance": 1e-7,
               "native_api": "qlib.backtest.backtest + SimulatorExecutor + Exchange + Position",
               "schedule_strategy": "DirectTargetWeightStrategy subclass with observed-open sizing and fee-affordable targets",
               "report_mark": "native daily same-day close; next-open reconciliation uses native positions and Qlib D.features",
               "scope": "prescribed open-only schedule settlement; controller selection and intraday recovery not independently validated",
               "execution_limitation": "Historical open marks, not executable 09:31 prices; continuous adjusted shares, no spread/impact/partial fills."}
    return {"audit": summary, "report": report, "positions": positions, "indicators": indicators,
            "reconciliation": reconciliation, "risk_analysis": risk_analysis(close_net, freq="day", mode="product")}
