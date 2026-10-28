"""Revalue frozen raw-share Stage1 holdings on complete Qlib minute marks."""
from __future__ import annotations

import numpy as np
import pandas as pd


def point_price_parity(raw_points: pd.DataFrame, quotes: pd.DataFrame) -> dict:
    stamps = pd.DatetimeIndex(raw_points.index)
    clock = stamps.strftime("%H:%M")
    opening = np.isin(clock, ["09:47", "10:32"])
    lookup = stamps + pd.to_timedelta(opening.astype(int), unit="min")
    observed = np.where(opening[:, None],
        quotes[["SOXL_open", "SOXX_open"]].reindex(lookup).to_numpy(float),
        quotes[["SOXL_close", "SOXX_close"]].reindex(lookup).to_numpy(float))
    expected = raw_points[["SOXL", "SOXX"]].to_numpy(float)
    if not np.isfinite(observed).all() or not np.array_equal(observed, expected):
        raise ValueError("blocked: Qlib full-minute raw prices differ from frozen account points")
    return {"point_rows": len(stamps), "max_absolute_error": 0.0}


def revalue(book: dict, quotes: pd.DataFrame, *, initial_cash: float) -> tuple[pd.DataFrame, dict]:
    authority = book["authority"].reset_index(drop=True)
    points = book["independent_points"].copy()
    points["stamp"] = pd.to_datetime(points.stamp)
    points["day"] = points.stamp.dt.normalize()
    points["clock"] = points.stamp.dt.strftime("%H:%M")
    trade_days = pd.DatetimeIndex(authority.trade_date)
    terminal_day = pd.Timestamp(authority.exit_date.iloc[-1])
    if len(authority) != 944 or book["audit"]["status"] != "passed":
        raise ValueError("blocked: frozen Stage1 raw account differs")
    states = {}
    for role, clock in (("pre", "09:45"), ("open", "09:47"), ("recovery", "10:32")):
        part = points.loc[points.clock.eq(clock) & points.day.isin(trade_days)]
        part = part.set_index("day")[["q_soxl", "q_soxx", "cash", "receivable"]]
        if len(part) != len(trade_days) or part.index.has_duplicates:
            raise ValueError("blocked: Stage1 minute position state incomplete")
        states[role] = part
    timestamps = pd.DatetimeIndex(quotes.index)
    used = (timestamps.normalize().isin(trade_days)
        & (timestamps >= trade_days[0] + pd.Timedelta(hours=9, minutes=47)))
    used |= (timestamps.normalize() == terminal_day) & (timestamps <= terminal_day + pd.Timedelta(hours=9, minutes=47))
    minute = quotes.loc[used].copy()
    timestamps = pd.DatetimeIndex(minute.index)
    day = timestamps.normalize()
    clock = timestamps.strftime("%H:%M")
    held = np.zeros((len(minute), 4), dtype=float)
    for role, mask in (("pre", clock <= "09:47"),
                       ("open", (clock >= "09:48") & (clock <= "10:32")),
                       ("recovery", clock >= "10:33")):
        active = mask & day.isin(trade_days)
        held[active] = states[role].reindex(day[active]).to_numpy(float)
    terminal_state = points.loc[points.day.eq(terminal_day) & points.clock.eq("09:45"),
        ["q_soxl", "q_soxx", "cash", "receivable"]]
    if len(terminal_state) != 1:
        raise ValueError("blocked: terminal pre-open state missing")
    held[day == terminal_day] = terminal_state.to_numpy(float)
    if not np.isfinite(held).all() or (held[:, :2] < 0).any() or (held[:, 2:] < -1e-7).any():
        raise ValueError("blocked: minute holdings/cash/receivable invalid")

    def marks(field: str) -> np.ndarray:
        prices = minute[["SOXL_" + field, "SOXX_" + field]].to_numpy(float)
        return np.sum(held[:, :2] * prices, axis=1) + held[:, 2] + held[:, 3]

    close_nav, low_nav, high_nav = marks("close"), marks("low"), marks("high")
    path = pd.DataFrame({"stamp": timestamps, "kind": "minute_close", "priority": 0,
        "nav": close_nav, "lower": low_nav, "upper": high_nav})
    if not np.isfinite(path[["nav", "lower", "upper"]]).all().all() or path.lower.le(0).any():
        raise ValueError("blocked: invalid frozen minute NAV")

    # Raw account points are exact cash/receivable/quantity marks. The 09:47
    # and 10:32 points are after fill; add a separate before-fill mark at the
    # corresponding bar open, then keep the original event NAV after fill.
    events = points.loc[points.day.isin(trade_days)
        | (points.day.eq(terminal_day) & points.clock.eq("09:45")),
        ["stamp", "clock", "nav"]].copy()
    events["kind"] = "original_event"
    events["priority"] = np.where(events.clock.isin(["09:47", "10:32"]), 2, 1)
    before = []
    for role, clock, prior in (("open", "09:47", "pre"), ("recovery", "10:32", "open")):
        at = trade_days + pd.Timedelta(hours=int(clock[:2]), minutes=int(clock[3:]))
        price = minute[["SOXL_open", "SOXX_open"]].reindex(at + pd.Timedelta(minutes=1)).to_numpy(float)
        state = states[prior].reindex(trade_days).to_numpy(float)
        values = np.sum(state[:, :2] * price, axis=1) + state[:, 2] + state[:, 3]
        after = points.loc[points.clock.eq(clock) & points.day.isin(trade_days), "nav"].to_numpy(float)
        if not np.isfinite(values).all() or not np.isfinite(after).all():
            raise ValueError("blocked: pre/post fill valuation incomplete")
        before.append(pd.DataFrame({"stamp": at, "kind": role + "_before_fill", "priority": 1,
                                    "nav": values, "lower": values, "upper": values}))
    events = events[["stamp", "kind", "priority", "nav"]]
    events["lower"], events["upper"] = events.nav, events.nav
    terminal = points.loc[points.day.eq(pd.Timestamp(authority.exit_date.iloc[-1]))
                          & points.clock.eq("09:47"), ["stamp", "nav"]]
    if len(terminal) != 1 or not np.isclose(float(terminal.nav.iloc[0]),
                                          float(authority.native_expected_nav.iloc[-1]), atol=1e-7):
        raise ValueError("blocked: frozen terminal mark differs")
    terminal = terminal.assign(kind="terminal_next_open", priority=2,
                               lower=terminal.nav, upper=terminal.nav)
    rth_days = trade_days[1:].append(pd.DatetimeIndex([terminal_day]))
    opening_at = rth_days + pd.Timedelta(hours=9, minutes=31)
    opening_prices = quotes[["SOXL_open", "SOXX_open"]].reindex(opening_at).to_numpy(float)
    opening_state = states["pre"].reindex(rth_days).to_numpy(float)
    opening_state[-1] = terminal_state.to_numpy(float)[0]
    opening_nav = np.sum(opening_state[:, :2] * opening_prices, axis=1) + opening_state[:, 2] + opening_state[:, 3]
    if not np.isfinite(opening_nav).all():
        raise ValueError("blocked: RTH opening gap mark missing")
    rth_open = pd.DataFrame({"stamp": rth_days + pd.Timedelta(hours=9, minutes=30),
        "kind": "rth_open", "priority": 0, "nav": opening_nav,
        "lower": opening_nav, "upper": opening_nav})
    start = pd.DataFrame([{"stamp": trade_days[0] + pd.Timedelta(hours=9, minutes=47),
        "kind": "initial_cash", "priority": -1, "nav": initial_cash,
        "lower": initial_cash, "upper": initial_cash}])
    path = pd.concat([path, events, *before, terminal, rth_open, start], ignore_index=True)
    path = path.sort_values(["stamp", "priority"], kind="stable").reset_index(drop=True)
    path["observed_peak"] = path.nav.cummax()
    path["upper_peak"] = path.upper.cummax()
    path["observed_drawdown"] = 1 - path.nav / path.observed_peak
    path["drawdown_upper_bound"] = 1 - path.lower / path.upper_peak
    if (path.observed_drawdown > path.drawdown_upper_bound + 1e-12).any():
        raise ValueError("blocked: OHLC envelope does not enclose observed account marks")
    # Every daily close is an exact frozen account point, including half days.
    actual = points.loc[points.day.isin(trade_days)].groupby("day").tail(1).set_index("day").nav
    observed = pd.DataFrame({"day": day, "nav": close_nav})
    observed = observed.loc[observed.day.isin(trade_days)].groupby("day").tail(1).set_index("day").nav
    close_error = (actual - observed).abs()
    if len(close_error) != len(trade_days) or close_error.max() > 1e-7:
        raise ValueError("blocked: minute close NAV differs from native Stage1 point")
    summary = {"trade_days": len(trade_days), "minute_rows": len(minute),
        "close_nav_max_error_usd": float(close_error.max()),
        "observed_drawdown": float(path.observed_drawdown.max()),
        "conservative_ohlc_bound": float(path.drawdown_upper_bound.max()),
        "observed_worst_at": str(path.loc[path.observed_drawdown.idxmax(), "stamp"]),
        "bound_worst_at": str(path.loc[path.drawdown_upper_bound.idxmax(), "stamp"]),
        "qualification": ("observed_source_mark_breach" if path.observed_drawdown.max() > .30
            else "bounded_RTH_source_risk_within30_under_assumptions" if path.drawdown_upper_bound.max() <= .30
            else "unresolved_intraminute_ordering")}
    return path, summary
