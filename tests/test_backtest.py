from __future__ import annotations

import pandas as pd

from src.soxl_quant.backtest import _normalize_backtest_seg
from src.soxl_quant.core.dataset import slice_with_lookback


def test_normalize_backtest_seg_clamps_terminal_calendar_date():
    index = pd.MultiIndex.from_product(
        [["SOXL"], pd.to_datetime(["2026-04-23", "2026-04-24", "2026-04-27"])],
        names=["instrument", "datetime"],
    )

    assert _normalize_backtest_seg(index, ("2026-04-23", "2026-04-27")) == (
        "2026-04-23",
        "2026-04-24",
    )


def test_normalize_backtest_seg_keeps_non_terminal_end_date():
    index = pd.MultiIndex.from_product(
        [["SOXL"], pd.to_datetime(["2026-04-23", "2026-04-24", "2026-04-27"])],
        names=["instrument", "datetime"],
    )

    assert _normalize_backtest_seg(index, ("2026-04-23", "2026-04-24")) == (
        "2026-04-23",
        "2026-04-24",
    )


def test_slice_with_lookback_keeps_prewindow_context() -> None:
    index = pd.bdate_range("2024-01-02", periods=8)
    frame = pd.DataFrame({"value": range(len(index))}, index=index)

    sliced = slice_with_lookback(frame, "2024-01-05", "2024-01-10", lookback=2)

    assert list(sliced.index) == list(index[1:7])


def test_slice_with_lookback_clamps_to_available_history() -> None:
    index = pd.bdate_range("2024-01-02", periods=4)
    frame = pd.DataFrame({"value": range(len(index))}, index=index)

    sliced = slice_with_lookback(frame, "2024-01-03", "2024-01-05", lookback=10)

    assert list(sliced.index) == list(index[:4])
