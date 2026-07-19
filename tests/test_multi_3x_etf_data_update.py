from __future__ import annotations

import pandas as pd


def _ohlcv(date: str, close: float) -> dict[str, object]:
    return {
        "date": date,
        "open": close - 1.0,
        "high": close + 1.0,
        "low": close - 2.0,
        "close": close,
        "volume": 1000,
    }


def test_pair_panel_to_qlib_frame_keeps_shared_vix_for_pair_provider():
    from src.data.multi_3x_etf_update import Multi3xEtfPairSpec, combine_pair_frames, pair_panel_to_qlib_frame

    spec = Multi3xEtfPairSpec("TQQQ", "QQQ")
    panel = combine_pair_frames(
        spec,
        etf=pd.DataFrame([_ohlcv("2026-07-01", 70.0)]),
        underlying=pd.DataFrame([_ohlcv("2026-07-01", 700.0)]),
        vix=pd.DataFrame([{"date": "2026-07-01", "vix": 17.0}]),
    )

    qlib_frame = pair_panel_to_qlib_frame(panel, spec)

    assert set(qlib_frame["instrument"]) == {"TQQQ", "QQQ"}
    assert set(qlib_frame["vix"]) == {17.0}
    assert {"datetime", "instrument", "open", "high", "low", "close", "volume", "vix"}.issubset(
        qlib_frame.columns
    )


def test_multi_3x_registry_contains_diversified_long_equity_pairs():
    from src.data.multi_3x_etf_update import MULTI_3X_ETF_PAIRS

    pairs = {(item.etf, item.underlying) for item in MULTI_3X_ETF_PAIRS}

    assert len(pairs) == 15
    assert {
        ("TNA", "IWM"),
        ("FAS", "XLF"),
        ("LABU", "XBI"),
        ("ERX", "XLE"),
        ("WANT", "XLY"),
    }.issubset(pairs)
    assert {"SQQQ", "SPXU"}.isdisjoint({etf for etf, _ in pairs})


def test_update_multi_3x_etf_data_appends_and_builds_provider(tmp_path, monkeypatch):
    from src.data import multi_3x_etf_update
    from src.data.multi_3x_etf_update import Multi3xEtfPairSpec, Multi3xEtfUpdatePaths

    spec = Multi3xEtfPairSpec("TQQQ", "QQQ")
    paths = Multi3xEtfUpdatePaths(
        panel_dir=tmp_path / "processed" / "panels",
        raw_symbol_dir=tmp_path / "raw" / "symbols",
        qlib_dir=tmp_path / "qlib",
        registry_yaml=tmp_path / "datasets.yaml",
    )
    existing = pd.DataFrame(
        [
            {
                "date": "2026-06-30",
                "tqqq_open": 60.0,
                "tqqq_high": 62.0,
                "tqqq_low": 59.0,
                "tqqq_close": 61.0,
                "tqqq_volume": 100,
                "qqq_open": 600.0,
                "qqq_high": 602.0,
                "qqq_low": 599.0,
                "qqq_close": 601.0,
                "qqq_volume": 200,
                "vix": 18.0,
            }
        ]
    )
    panel_path = paths.panel_dir / "tqqq_with_underlying_vix.csv"
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    existing.to_csv(panel_path, index=False)

    def fake_fetch(symbol: str, *, start: str, end: str) -> pd.DataFrame:
        assert start == "2026-06-29"
        assert end == "2026-07-02"
        close = 70.0 if symbol == "TQQQ" else 700.0
        return pd.DataFrame([_ohlcv("2026-07-02", close)])

    def fake_vix(*, start: str, end: str):
        return pd.DataFrame([{"date": "2026-07-02", "vix": 16.0}]), "fake_vix"

    monkeypatch.setattr(multi_3x_etf_update, "_fetch_yfinance_compatible_ohlcv", fake_fetch)
    monkeypatch.setattr(multi_3x_etf_update, "_fetch_vix_with_fallback", fake_vix)

    result = multi_3x_etf_update.update_multi_3x_etf_data(
        paths=paths,
        pairs=[spec],
        end="2026-07-02",
        refresh_overlap_days=1,
    )

    updated = pd.read_csv(panel_path)
    raw_symbol = pd.read_csv(paths.raw_symbol_dir / "TQQQ.csv")
    assert result["pair_results"][0]["new_last_date"] == "2026-07-02"
    assert result["pair_results"][0]["appended_rows"] == 1
    assert updated["date"].tolist() == ["2026-06-30", "2026-07-02"]
    assert raw_symbol["date"].tolist() == ["2026-06-30", "2026-07-02"]
    assert (paths.qlib_dir / "tqqq" / "qlib_store_manifest.json").exists()
    assert result["registry"]["updated_keys"] == [
        "processed_panels.tqqq_with_underlying_vix",
        "symbols.TQQQ",
        "qlib.TQQQ",
    ]
