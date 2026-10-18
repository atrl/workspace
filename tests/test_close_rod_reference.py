from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.backtest.close_rod_reference import passive_expand_native_book


def test_passive_marks_preserve_native_state_and_reject_missing_or_changed_anchors():
    day, half, terminal = pd.to_datetime(["2023-01-04", "2023-01-05", "2023-01-06"])
    stamps = [d + pd.Timedelta(clock) for d, clocks in (
        (day, ["9h45min", "9h47min", "10h32min", "16h"]),
        (half, ["9h45min", "9h47min", "10h32min", "13h"]),
        (terminal, ["9h45min", "9h47min"])) for clock in clocks]
    old_quotes = pd.DataFrame({"SOXL": [10., 10., 10., 12., 12., 12., 12., 11., 6., 6.],
                               "SOXX": [20., 20., 20., 21., 21., 21., 21., 22., 22., 22.]},
                              index=pd.DatetimeIndex(stamps))
    states = np.array([[0, 0, 100, 0], [2, 1, 58, 2], [3, 1, 48, 2], [3, 1, 48, 2],
                       [3, 1, 50, 0], [3, 1, 50, 0], [3, 1, 50, 0], [3, 1, 50, 0],
                       [6, 1, 50, 0], [6, 1, 50, 0]], dtype=float)
    points = pd.DataFrame(states, columns=["q_soxl", "q_soxx", "cash", "receivable"])
    points.insert(0, "stamp", stamps)
    points["nav"] = (states[:, :2] * old_quotes.to_numpy()).sum(axis=1) + states[:, 2:].sum(axis=1)
    points["relative_error"] = 0.
    orders = pd.DataFrame({"stamp": [stamps[1], stamps[1], stamps[2]], "raw_shares": [2, 1, 1],
                           "total": [.1, .2, .1]})  # Multiple orders at one native post-event stamp.
    corporate = pd.DataFrame({"date": [half, terminal], "kind": ["pay", "split"]})
    book = {"points": points, "orders": orders, "native_orders": orders.copy(),
            "authority": pd.DataFrame({"trade_date": [day, half], "exit_date": [half, terminal],
                                       "strategy_return": [.03, -.01], "equity": [1.03, 1.0197]}),
            "native_report": pd.DataFrame({"account": points.nav.to_numpy()}, index=old_quotes.index),
            "native_positions": {stamp: {"synthetic_frozen_marker": i} for i, stamp in enumerate(stamps)},
            "corporate_receipts": corporate, "native_corporate_receipts": corporate.copy(),
            "reconciliation": points.copy(), "audit": {"status": "passed", "native_CA_from_own_positions": True,
                                                        "points": len(points)},
            "independent_points": pd.concat([points.iloc[[1]], points], ignore_index=True)}
    original = deepcopy(book)
    added_stamps = pd.DatetimeIndex([day + pd.Timedelta(x) for x in ["15h30min", "15h32min", "15h59min"]])
    added_quotes = pd.DataFrame({"SOXL": [11., 12., 9.], "SOXX": [22., 23., 24.]}, index=added_stamps)
    quotes = pd.concat([old_quotes, added_quotes]).sort_index()
    identity = {"original_point_prices": old_quotes, "recorder_id": "synthetic-frozen-reference"}
    expanded = passive_expand_native_book(book, point_prices=quotes, source_identity=identity)
    derived = expanded["passive_added_points"]
    np.testing.assert_array_equal(derived[["q_soxl", "q_soxx", "cash", "receivable"]], [[3, 1, 48, 2]] * 3)
    np.testing.assert_array_equal(derived.nav, [105., 109., 101.])
    assert derived.relative_error.isna().all() and not derived.native_engine_validated.any()
    assert expanded["points"].iloc[-1].stamp == stamps[-1]
    assert len(expanded["points"]) == len(points) + 3
    assert expanded["passive_expansion_audit"]["new_native_accounts"] == 0
    for key in ["points", "orders", "native_orders", "authority", "native_report", "corporate_receipts",
                "native_corporate_receipts", "reconciliation", "independent_points"]:
        pd.testing.assert_frame_equal(book[key], original[key], check_exact=True)
        if key != "points":
            pd.testing.assert_frame_equal(expanded[key], original[key], check_exact=True)
    pd.testing.assert_frame_equal(expanded["original_native_points"], points, check_exact=True)
    assert expanded["audit"] == book["audit"] == original["audit"]
    assert expanded["native_positions"] == original["native_positions"]

    for bad in [quotes.drop(added_stamps[1]),
                pd.concat([quotes, pd.DataFrame([[1., 1.]], index=[terminal + pd.Timedelta("15h30min")], columns=quotes.columns)]).sort_index(),
                quotes.assign(SOXL=quotes.SOXL.mask(quotes.index == added_stamps[0], np.nan))]:
        with pytest.raises(ValueError):
            passive_expand_native_book(book, point_prices=bad, source_identity=identity)
    changed = quotes.copy()
    changed.loc[stamps[0], "SOXL"] += 1  # Unheld asset: NAV-only matching would miss this.
    with pytest.raises(ValueError, match="two-asset raw prices changed"):
        passive_expand_native_book(book, point_prices=changed, source_identity=identity)
    unknown = deepcopy(book)
    unknown["orders"].loc[0, "stamp"] = added_stamps[0]
    with pytest.raises(ValueError, match="intervening event"):
        passive_expand_native_book(unknown, point_prices=quotes, source_identity=identity)
    ambiguous = deepcopy(book)
    ambiguous["points"] = pd.concat([points.iloc[[0]], points], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        passive_expand_native_book(ambiguous, point_prices=quotes, source_identity=identity)
