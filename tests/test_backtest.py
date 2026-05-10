from __future__ import annotations

import pandas as pd

from src.soxl_quant.backtest import _normalize_backtest_seg


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
