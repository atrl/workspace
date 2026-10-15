"""Post-session native accounting replay, with a declared logical event axis.

Actual receipt and per-symbol price times stay in ``price_events``. Logical
labels order accounting transitions; they are not fabricated market timestamps.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import IBKRTieredExchange
from src.backtest.native_raw_corporate import RawCashPosition
from src.data.diagnostic_forward_inputs import digest

SYMBOLS = ("SOXL", "SOXX")
PRICE_FIELDS = ("close", "execution_price", "raw_execution_price")


def _events(price_events):
    events = pd.DataFrame(price_events).copy()
    if "received_at" not in events and "receipt" in events:
        events["received_at"] = events.receipt.map(lambda x: x["received"]["utc"])
    required = {"event_id", "logical_stamp", "session", "price_soxl", "price_soxx",
                "ca_due", "endpoint_role", "received_at", "price_times"}
    if not required.issubset(events):
        raise ValueError("native replay needs complete event identities and receipt times")
    events["logical_stamp"] = pd.to_datetime(events.logical_stamp)
    if events.logical_stamp.dt.tz is not None:
        events["logical_stamp"] = events.logical_stamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    if events.event_id.duplicated().any() or events.logical_stamp.duplicated().any():
        raise ValueError("duplicate native event")
    if not events.logical_stamp.is_monotonic_increasing:
        raise ValueError("native events out of recorded order")
    if not events.logical_stamp.dt.date.equals(pd.to_datetime(events.session).dt.date):
        raise ValueError("logical labels must retain real fee/corporate session dates")
    values = events[["price_soxl", "price_soxx"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("unknown marks cannot become native observations")
    receipts = pd.to_datetime(events.received_at, utc=True)
    if receipts.isna().any() or not receipts.is_monotonic_increasing:
        raise ValueError("native replay needs chronological actual receipts")
    for times, received in zip(events.price_times, receipts):
        if set(times) != set(SYMBOLS):
            raise ValueError("missing original per-symbol price times")
        for value in times.values():
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None or pd.isna(stamp) or stamp.tz_convert("UTC") > received:
                raise ValueError("invalid original price timestamp")
    if (events.endpoint_role == "start").sum() != 1 or (events.endpoint_role == "terminal").sum() != 1:
        raise ValueError("exactly one pretrade starting and terminal endpoint required")
    if events.iloc[-1].endpoint_role != "terminal":
        raise ValueError("terminal must be last observed event")
    return events


def build_event_provider(price_events, provider_uri):
    """Build and read back a sparse accounting grid outside an active Recorder."""
    from qlib.data import D
    from src.data.qlib_native import build_qlib_store_from_frame

    events = _events(price_events)
    guard = events.logical_stamp.iloc[-1] + pd.Timedelta(minutes=1)
    rows = []
    for event in events.itertuples():
        for symbol, price in zip(SYMBOLS, (event.price_soxl, event.price_soxx)):
            rows.append({"instrument": symbol, "datetime": event.logical_stamp,
                         **dict.fromkeys(PRICE_FIELDS, float(price)), "factor": 1.0})
    # Required next calendar bound only. Never included in backtest or metrics.
    for symbol, price in zip(SYMBOLS, events.iloc[-1][["price_soxl", "price_soxx"]]):
        rows.append({"instrument": symbol, "datetime": guard,
                     **dict.fromkeys(PRICE_FIELDS, float(price)), "factor": 1.0})
    source = pd.DataFrame(rows)
    manifest = build_qlib_store_from_frame(
        source, provider_uri=Path(provider_uri), freq="1min", region="us",
        as_of=str(guard.date()), field_map={x: x for x in (*PRICE_FIELDS, "factor")},
        source="Recorded paper-event accounting axis; actual price/receipt times in event map; final row boundary only",
    )
    calendar = pd.DatetimeIndex(D.calendar(start_time=events.logical_stamp.iloc[0], end_time=guard, freq="1min"))
    expected = pd.DatetimeIndex([*events.logical_stamp, guard])
    pd.testing.assert_index_equal(calendar.rename(None), expected.rename(None))
    readback = D.features(list(SYMBOLS), ["$" + x for x in PRICE_FIELDS],
                          start_time=calendar[0], end_time=calendar[-1], freq="1min")
    for symbol in SYMBOLS:
        raw = source[source.instrument.eq(symbol)].set_index("datetime")[list(PRICE_FIELDS)]
        got = readback.xs(symbol, level="instrument")
        pd.testing.assert_index_equal(got.index.rename(None), raw.index.rename(None))
        np.testing.assert_array_equal(got.to_numpy(), raw.to_numpy(dtype=np.float32))
    return {"manifest": manifest, "provider_uri": str(Path(provider_uri).resolve()),
            "events": events, "events_sha256": digest(events),
            "float32_readback": readback, "boundary_only_guard": str(guard),
            "guard_excluded_from_backtest": True,
            "logical_axis_not_market_clock": True}


def validate_event_book(points, orders, price_events, provider_uri, schedule, tariff, cost, initial_cash=100000.0):
    """Reconcile recorded quantities against independently owned native state."""
    import qlib
    from qlib.backtest import backtest
    from qlib.backtest.decision import Order, TradeDecisionWO
    from qlib.backtest.high_performance_ds import NumpyQuote
    from src.backtest.strategy import DirectTargetWeightStrategy

    events = _events(price_events)
    points, orders = pd.DataFrame(points).copy(), pd.DataFrame(orders).copy()
    if points.event_id.duplicated().any() or list(points.event_id) != list(events.event_id):
        raise ValueError("expected book points differ from full recorded event grid")
    points = points.set_index("event_id")
    if len(orders) and not set(orders.event_id).issubset(set(events.event_id)):
        raise ValueError("order outside recorded event grid")
    if len(orders):
        terminal_id = events.loc[events.endpoint_role.eq("terminal"), "event_id"].iloc[0]
        if orders.event_id.eq(terminal_id).any():
            raise ValueError("terminal is a pretrade endpoint, not a trading slot")
        shares = orders.raw_shares.to_numpy(float)
        if not np.isfinite(shares).all() or (shares <= 0).any() or (shares != np.floor(shares)).any():
            raise ValueError("native orders require positive whole raw shares")
        if not set(orders.symbol).issubset(SYMBOLS):
            raise ValueError("unexpected traded asset")
    qlib.init(provider_uri=str(provider_uri), region="us", kernels=1)
    by_time = events.set_index("logical_stamp")
    scheduled = dict(tuple(orders.groupby("event_id", sort=False))) if len(orders) else {}
    exact_prices = {(symbol, row.logical_stamp): float(price)
                    for row in events.itertuples()
                    for symbol, price in zip(SYMBOLS, (row.price_soxl, row.price_soxx))}
    overlay_audit = []

    class ReceiptQuote(NumpyQuote):
        def __init__(self, quote_df, freq):
            exact = quote_df.copy()
            for name in PRICE_FIELDS:
                exact["$" + name] = exact["$" + name].astype("float64")
            for (symbol, stamp), price in exact_prices.items():
                prior = float(exact.loc[(symbol, stamp), "$close"])
                if prior != float(np.float32(price)):
                    raise ValueError("Qlib quote differs from registered float32 receipt value")
                exact.loc[(symbol, stamp), ["$" + x for x in PRICE_FIELDS]] = price
                overlay_audit.append({"symbol": symbol, "logical_stamp": stamp,
                                      "qlib_float32": prior, "receipt_float64": price})
            super().__init__(exact, freq, region="us")

    endpoints, corporate_receipts = {}, []

    class RecordedOrders(DirectTargetWeightStrategy):
        def __init__(self):
            super().__init__(signal=pd.Series(dtype=float, index=pd.MultiIndex.from_arrays(
                [[], []], names=["datetime", "instrument"])), risk_degree=1.0)

        def generate_trade_decision(self, execute_result=None):
            start, end = self.trade_calendar.get_step_time(self.trade_calendar.get_trade_step())
            event = by_time.loc[pd.Timestamp(start)]
            position = self.trade_position
            if event.ca_due:
                corporate_receipts.extend(position.apply_corporate(pd.Timestamp(event.session), schedule))
            if event.endpoint_role in ("start", "terminal"):
                q = np.array([position.get_stock_amount(s) if position.check_stock(s) else 0.0 for s in SYMBOLS])
                endpoints[event.endpoint_role] = float(q @ event[["price_soxl", "price_soxx"]].to_numpy(float)
                                                       + position.get_cash() + position.income.value)
            records = scheduled.get(event.event_id)
            submitted = [] if records is None else [Order(
                str(row.symbol), float(row.raw_shares), Order.SELL if row.sell else Order.BUY, start, end
            ) for row in records.itertuples()]
            return TradeDecisionWO(submitted, self)

    first, last = events.logical_stamp.iloc[0], events.logical_stamp.iloc[-1]
    exchange = IBKRTieredExchange(tariff=tariff, slippage_bps=cost, freq="1min",
                                  start_time=first, end_time=last, codes=list(SYMBOLS),
                                  deal_price="execution_price", limit_threshold=None, quote_cls=ReceiptQuote)
    outputs, indicators = backtest(
        start_time=first, end_time=last, strategy=RecordedOrders(),
        executor={"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                  "kwargs": {"time_per_step": "1min", "generate_portfolio_metrics": True}},
        account=initial_cash, benchmark=pd.Series(0.0, index=pd.DatetimeIndex(events.logical_stamp)),
        pos_type=RawCashPosition, exchange_kwargs={"exchange": exchange},
    )
    report, positions = outputs["1min"]
    if set(positions) != set(events.logical_stamp):
        raise ValueError("native grid differs, or boundary guard entered valuation")
    checks = []
    for stamp, position in positions.items():
        event = by_time.loc[stamp]
        expected = points.loc[event.event_id]
        q = np.array([position.get_stock_amount(s) if position.check_stock(s) else 0.0 for s in SYMBOLS])
        actual = np.r_[q, position.get_cash(), position.income.value, position.calculate_value()]
        wanted = expected[["q_soxl", "q_soxx", "cash", "receivable", "nav"]].to_numpy(float)
        if not np.isfinite(actual).all() or not np.isfinite(wanted).all():
            raise ValueError("unknown NAV or holdings cannot pass reconciliation")
        np.testing.assert_allclose(actual, wanted, rtol=1e-10, atol=1e-7)
        checks.append({"event_id": event.event_id, "logical_stamp": stamp,
                       "maximum_absolute_error": float(np.abs(actual - wanted).max())})
    native_fees = pd.DataFrame(exchange.executed_fees)
    if len(native_fees) != len(orders):
        raise ValueError("native order count mismatch")
    if len(orders):
        np.testing.assert_allclose(native_fees.total, orders.total, rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.cumprod(1 + report["return"] - report["cost"]) * initial_cash,
                               report.account, rtol=1e-9, atol=1e-7)
    return {"status": "passed", "passed": True, "reconciliation": pd.DataFrame(checks), "native_report": report,
            "native_positions": positions, "native_orders": native_fees, "native_indicators": indicators,
            "corporate_receipts": corporate_receipts, "pretrade_endpoints": endpoints,
            "interval_return": endpoints["terminal"] / endpoints["start"] - 1,
            "receipt_precision_overlay": pd.DataFrame(overlay_audit),
            "logical_grid_only": True, "actual_clock_map": events,
            "annualized_risk_from_this_grid": False, "independent_CA_state": True}
