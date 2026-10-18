"""Passive late marks for an already validated native raw-share account.

No strategy, execution engine or corporate-action ledger runs here.  The original
native account remains the authority for quantities, cash, receivables, fees and
daily returns.  New marks have a separate passive evidence role.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd


_ASSETS = ("SOXL", "SOXX")
_STATE = ("q_soxl", "q_soxx", "cash", "receivable")
_MORNING = ("09:45", "09:47", "10:32")
_LATE = ("15:30", "15:32", "15:59")


def _require(condition, message):
    if not condition:
        raise ValueError("needs_data: passive reference " + message)


def _index(values, name):
    index = pd.DatetimeIndex(values)
    _require(len(index) > 0 and not index.hasnans and index.tz is None
             and not index.has_duplicates and index.is_monotonic_increasing,
             name + " must be unique, sorted, naive exchange-local timestamps")
    return index


def _at(day, clock):
    return day + pd.Timedelta(hours=int(clock[:2]), minutes=int(clock[3:]))


def _quotes(frame, name):
    _require(isinstance(frame, pd.DataFrame) and list(frame.columns) == list(_ASSETS),
             name + " needs exactly SOXL/SOXX raw quote columns")
    index = _index(frame.index, name)
    values = frame.to_numpy(dtype=float)
    _require(np.isfinite(values).all() and (values > 0).all(), name + " has missing/invalid prices")
    return index


def passive_expand_native_book(book, *, point_prices, source_identity=None):
    """Add the three fixed late marks while preserving the old native account.

    ``source_identity`` must contain ``original_point_prices``, the hash-bound
    original two-asset quote DataFrame.  Scalar lineage metadata may accompany
    it.  Quantities or NAV alone cannot establish unchanged quotes for an asset
    with zero holdings.  Both quote grids must have been read through Qlib by the
    caller.  This routine performs no source ingestion or market-data reads.

    The supported frozen account has unique post-event native ``points`` at
    09:45/09:47/10:32 and its actual 13:00 or 16:00 close, followed by the terminal
    day's 09:45/09:47.  Duplicate orders at a stamp are retained; ambiguous native
    point duplicates are rejected rather than resolved by selecting a last row.
    """
    _require(isinstance(book, dict) and "passive_expansion_audit" not in book,
             "requires an unexpanded frozen book")
    _require(book.get("audit", {}).get("status") == "passed"
             and book["audit"].get("native_CA_from_own_positions") is True,
             "has no original passed native raw-account audit")
    required = {"points", "authority", "orders", "native_orders", "native_report", "native_positions",
                "corporate_receipts", "native_corporate_receipts", "reconciliation"}
    _require(required <= set(book), "original native evidence incomplete")
    _require(isinstance(source_identity, dict) and "original_point_prices" in source_identity,
             "requires original two-asset price anchors, not NAV-only matching")
    original_quotes = source_identity["original_point_prices"]
    old_quote_index = _quotes(original_quotes, "original quotes")
    quote_index = _quotes(point_prices, "expanded quotes")
    points = book["points"]
    _require(isinstance(points, pd.DataFrame) and {"stamp", "nav", *_STATE} <= set(points),
             "native point schema incomplete")
    old_index = _index(points.stamp, "native points")
    _require(old_index.equals(old_quote_index), "original quote and native point grids differ")
    state_values = points.loc[:, list(_STATE)].to_numpy(dtype=float)
    nav_values = points.nav.to_numpy(dtype=float)
    _require(np.isfinite(state_values).all() and np.isfinite(nav_values).all()
             and (state_values >= -1e-7).all() and (nav_values > 0).all(), "invalid original q/C/R/NAV")
    _require(old_index.isin(quote_index).all(), "expanded grid omits original anchors")
    _require(np.array_equal(point_prices.loc[old_index].to_numpy(), original_quotes.to_numpy()),
             "old two-asset raw prices changed")
    old_recomputed = (state_values[:, :2] * original_quotes.to_numpy(dtype=float)).sum(axis=1) + state_values[:, 2:].sum(axis=1)
    _require(np.allclose(old_recomputed, nav_values, rtol=1e-7, atol=1e-7),
             "old raw quote/state basis differs from native NAV")

    authority = book["authority"]
    _require(isinstance(authority, pd.DataFrame) and {"trade_date", "exit_date"} <= set(authority),
             "daily authority dates absent")
    days = _index(authority.trade_date, "daily trade dates")
    exits = _index(authority.exit_date, "daily exit dates")
    _require(days.equals(days.normalize()) and exits.equals(exits.normalize())
             and len(days) == len(exits) and exits[:-1].equals(days[1:]) and exits[-1] > days[-1],
             "daily native session chain differs")
    terminal = exits[-1]
    by_stamp = points.set_index("stamp", drop=False)
    expected_old, additions, regular_days, half_days = [], [], [], []
    for day in days:
        observed = old_index[old_index.normalize() == day]
        clocks = observed.strftime("%H:%M").tolist()
        _require(len(observed) == 4 and clocks[:3] == list(_MORNING)
                 and clocks[-1] in ("13:00", "16:00"), "original daily point schedule changed")
        expected_old.extend(observed)
        if clocks[-1] == "16:00":
            regular_days.append(day)
            additions.extend(_at(day, clock) for clock in _LATE)
            before, closing = by_stamp.loc[_at(day, "10:32")], by_stamp.loc[_at(day, "16:00")]
            _require(np.array_equal(before[list(_STATE)].to_numpy(dtype=float),
                                    closing[list(_STATE)].to_numpy(dtype=float)),
                     "unexplained late quantity/cash/receivable change")
        else:
            half_days.append(day)
    expected_old.extend([_at(terminal, "09:45"), _at(terminal, "09:47")])
    _require(old_index.equals(pd.DatetimeIndex(expected_old)), "old dates/terminal boundary differ")
    new_index = pd.DatetimeIndex(additions)
    expected_expanded = old_index.union(new_index).sort_values()
    _require(quote_index.equals(expected_expanded),
             "expanded grid must contain exactly every fixed late anchor and no half-day/terminal/future additions")

    # The old engine applied all date-only corporate events before the first
    # 09:45 native snapshot.  Only the same day's post-10:32 state is carried;
    # no overnight fill-forward or unknown intraday event is inferred.
    for key in ("orders", "native_orders"):
        orders = book[key]
        _require(isinstance(orders, pd.DataFrame), key + " is not a native table")
        if len(orders):
            _require("stamp" in orders, key + " stamp missing")
            stamps = pd.DatetimeIndex(orders.stamp)
            _require(not stamps.hasnans and stamps.tz is None and stamps.isin(old_index).all()
                     and set(stamps.strftime("%H:%M")) <= {"09:47", "10:32"}
                     and stamps.normalize().isin(days).all(), key + " contains an unsupported intervening event")
    for key in ("corporate_receipts", "native_corporate_receipts"):
        receipts = book[key]
        _require(isinstance(receipts, pd.DataFrame), key + " is not a native table")
        if len(receipts):
            _require({"date", "kind"} <= set(receipts) and "stamp" not in receipts,
                     key + " event-clock schema differs from original date-only contract")
            dates = pd.DatetimeIndex(receipts.date)
            _require(not dates.hasnans and dates.tz is None and dates.equals(dates.normalize())
                     and dates.isin(days.append(pd.DatetimeIndex([terminal]))).all()
                     and set(receipts.kind) <= {"split", "entitle", "pay"},
                     key + " contains unknown corporate events")

    rows = []
    for stamp in new_index:
        held = by_stamp.loc[_at(stamp.normalize(), "10:32")]
        prices = point_prices.loc[stamp, list(_ASSETS)].to_numpy(dtype=float)
        # Preserve cash and receivables separately: receivable is value, not
        # funding cash.  No ledger/payment/fee state advances at a passive mark.
        state = held[list(_STATE)].to_numpy(dtype=float)
        nav = float(state[:2] @ prices + state[2] + state[3])
        row = {name: np.nan for name in points.columns}
        row.update({"stamp": stamp, "nav": nav, **dict(zip(_STATE, state)),
                    "valuation_role": "passive_expanded_raw_mark",
                    "native_engine_validated": False})
        rows.append(row)
    added = pd.DataFrame(rows, columns=[*points.columns, "valuation_role", "native_engine_validated"])
    original = points.copy(deep=True)
    original["valuation_role"] = "original_native_Qlib_point"
    original["native_engine_validated"] = True
    expanded = pd.concat([original, added], ignore_index=True).sort_values("stamp", kind="stable").reset_index(drop=True)
    _require(pd.DatetimeIndex(expanded.stamp).equals(quote_index), "expanded point construction lost an anchor")
    pd.testing.assert_frame_equal(expanded.loc[expanded.native_engine_validated, points.columns].reset_index(drop=True),
                                  points.reset_index(drop=True), check_exact=True)
    expanded.attrs.update(evidence_role="original_native_points_plus_passive_raw_marks",
                          new_points_native_engine_validated=False)

    result = dict(book)  # All original native objects remain unmodified.
    result["original_native_points"] = points.copy(deep=True)
    result["original_native_audit"] = deepcopy(book["audit"])
    result["points"] = expanded
    result["passive_added_points"] = added
    result["passive_expansion_audit"] = {
        "status": "passed", "role": "passive_frozen_holdings_valuation_not_native_execution",
        "original_native_audit_role": "unchanged_original_grid_orders_account_and_daily_returns_only",
        "original_native_points": len(points), "expanded_points": len(expanded), "added_passive_points": len(added),
        "regular_sessions": len(regular_days), "half_day_sessions": len(half_days),
        "terminal": str(old_index[-1]), "terminal_extensions": 0,
        "original_two_asset_quotes_exact": True, "original_point_values_exact": True,
        "original_orders_fees_corporate_and_daily_authority_unchanged": True,
        "new_points_native_engine_validated": False, "new_native_accounts": 0, "new_policy_replays": 0,
        "passive_reference_revaluations": 1,
        "holding_source": "same-session original post-event 10:32 native q/cash/receivable; unchanged through original close",
        "price_source": {k: v for k, v in source_identity.items() if k != "original_point_prices"},
        "scope": "Expanded sparse observed raw marks only; no fullminute/allhours/broker-fill risk qualification.",
    }
    result["evidence_roles"] = {
        "audit": "original_native_only; does not validate the added points",
        "native_report_positions_reconciliation": "unchanged_original_native_grid",
        "authority": "unchanged_original_native_daily_returns_and_terminal_wealth",
        "points": "original_native_points plus explicitly flagged passive marks",
        "passive_expansion_audit": "separate arithmetic/lineage validation; no native engine invocation",
    }
    return result
