from soxl_backtest import SOXLBacktest


def test_legacy_soxl_backtest_reads_current_local_data_paths_for_2026_window():
    backtest = SOXLBacktest(
        {
            "USE_LONGBRIDGE": False,
            "START_DATE": "2026-01-01",
            "END_DATE": "2026-06-09",
        }
    )

    assert not backtest.df.empty
    assert backtest.df["date"].min().date().isoformat() == "2026-01-02"
    assert backtest.df["date"].max().date().isoformat() == "2026-06-09"
    assert {"date", "price", "vix", "panic_score"}.issubset(backtest.df.columns)
