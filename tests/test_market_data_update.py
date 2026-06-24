from __future__ import annotations

from pathlib import Path

import pandas as pd


def _panel(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _ohlcv(date: str, close: float, *, volume: int = 1000) -> dict[str, object]:
    return {
        "date": date,
        "open": close - 1.0,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": volume,
    }


def test_market_update_preserves_existing_overlap_and_appends_new_rows(tmp_path, monkeypatch):
    from src.data import market_update

    panel_path = tmp_path / "panel.csv"
    paths = market_update.MarketDataUpdatePaths(
        panel_csv=panel_path,
        raw_symbol_dir=tmp_path / "raw" / "symbols",
        raw_external_dir=tmp_path / "raw" / "external",
        qlib_provider_uri=tmp_path / "qlib" / "soxl",
        registry_yaml=tmp_path / "datasets.yaml",
    )
    panel = _panel(
        [
            {
                "date": "2026-06-10",
                "soxl_open": 179.0,
                "soxl_high": 183.0,
                "soxl_low": 177.0,
                "soxl_close": 180.0,
                "soxl_volume": 10,
                "soxx_open": 539.0,
                "soxx_high": 543.0,
                "soxx_low": 537.0,
                "soxx_close": 540.0,
                "soxx_volume": 20,
                "vix": 18.0,
            }
        ]
    )
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(panel_path, index=False)

    def fake_fetch_market_frames(*, provider, start, end, **kwargs):
        assert provider == "futu"
        return market_update.MarketDataFrames(
            provider="futu",
            source_api=["futu.OpenQuoteContext.request_history_kline", "yahoo.chart"],
            soxl=_panel([_ohlcv("2026-06-10", 999.0), _ohlcv("2026-06-11", 223.99)]),
            soxx=_panel([_ohlcv("2026-06-10", 888.0), _ohlcv("2026-06-11", 586.65)]),
            vix=_panel(
                [
                    {"date": "2026-06-10", "vix": 18.2},
                    {"date": "2026-06-11", "vix": 17.9},
                ]
            ),
        )

    monkeypatch.setattr(market_update, "fetch_market_frames", fake_fetch_market_frames)

    result = market_update.update_market_data(
        paths=paths,
        provider="auto",
        provider_sequence=["futu", "yahoo_api"],
        end="2026-06-12",
        skip_qlib=True,
        skip_registry=True,
    )

    updated = pd.read_csv(panel_path)
    assert result["provider"] == "futu"
    assert result["old_last_date"] == "2026-06-10"
    assert result["new_last_date"] == "2026-06-11"
    assert result["appended_rows"] == 1
    assert updated.loc[updated["date"] == "2026-06-10", "soxl_close"].item() == 180.0
    assert updated.loc[updated["date"] == "2026-06-11", "soxl_close"].item() == 223.99


def test_local_surface_sync_derives_raw_symbol_and_vix_files(tmp_path):
    from src.data import market_update

    paths = market_update.MarketDataUpdatePaths(
        panel_csv=tmp_path / "processed" / "panels" / "soxl_soxx_vix_full.csv",
        raw_symbol_dir=tmp_path / "raw" / "symbols",
        raw_external_dir=tmp_path / "raw" / "external",
        qlib_provider_uri=tmp_path / "qlib" / "soxl",
        registry_yaml=tmp_path / "datasets.yaml",
    )
    panel = _panel(
        [
            {
                "date": "2026-06-11",
                "soxl_open": 221.0,
                "soxl_high": 226.0,
                "soxl_low": 219.0,
                "soxl_close": 223.99,
                "soxl_volume": 100,
                "soxx_open": 580.0,
                "soxx_high": 590.0,
                "soxx_low": 575.0,
                "soxx_close": 586.65,
                "soxx_volume": 200,
                "vix": 17.9,
            }
        ]
    )

    written = market_update.sync_local_data_surfaces(panel, paths=paths)

    assert written["raw_symbols"]["SOXL"].name == "SOXL.csv"
    soxl = pd.read_csv(paths.raw_symbol_dir / "SOXL.csv")
    vix = pd.read_csv(paths.raw_external_dir / "VIX.csv")
    assert list(soxl.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert soxl.loc[0, "close"] == 223.99
    assert vix.to_dict("records") == [{"date": "2026-06-11", "vix": 17.9}]


def test_market_panel_to_qlib_frame_includes_vix_for_each_instrument():
    from src.data import market_update

    panel = _panel(
        [
            {
                "date": "2026-06-11",
                "soxl_open": 221.0,
                "soxl_high": 226.0,
                "soxl_low": 219.0,
                "soxl_close": 223.99,
                "soxl_volume": 100,
                "soxx_open": 580.0,
                "soxx_high": 590.0,
                "soxx_low": 575.0,
                "soxx_close": 586.65,
                "soxx_volume": 200,
                "vix": 17.9,
            }
        ]
    )

    qlib_frame = market_update.market_panel_to_qlib_frame(panel)

    assert set(qlib_frame["instrument"]) == {"SOXL", "SOXX"}
    assert set(qlib_frame["vix"]) == {17.9}
    assert set(["datetime", "instrument", "open", "high", "low", "close", "volume", "vix"]).issubset(
        qlib_frame.columns
    )


def test_provider_raw_sync_appends_without_rewriting_existing_rows(tmp_path):
    from src.data import market_update

    paths = market_update.MarketDataUpdatePaths(
        panel_csv=tmp_path / "panel.csv",
        raw_symbol_dir=tmp_path / "raw" / "symbols",
        raw_external_dir=tmp_path / "raw" / "external",
        qlib_provider_uri=tmp_path / "qlib" / "soxl",
        registry_yaml=tmp_path / "datasets.yaml",
    )
    existing_path = paths.raw_external_dir / "futu_history_kline_US_SOXL.csv"
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_text = (
        "symbol,date,code,time_key,close\n"
        "US.SOXL,2026-06-10,US.SOXL,2026-06-10 00:00:00,180.6500000000000001\n"
    )
    existing_path.write_text(existing_text, encoding="utf-8")
    frames = market_update.MarketDataFrames(
        provider="futu",
        source_api=["futu"],
        soxl=pd.DataFrame(),
        soxx=pd.DataFrame(),
        vix=pd.DataFrame(),
        raw_provider_frames={
            "futu_history_kline_US_SOXL": pd.DataFrame(
                [
                    {
                        "symbol": "US.SOXL",
                        "date": "2026-06-10",
                        "code": "US.SOXL",
                        "time_key": "2026-06-10 00:00:00",
                        "close": 180.65,
                    },
                    {
                        "symbol": "US.SOXL",
                        "date": "2026-06-11",
                        "code": "US.SOXL",
                        "time_key": "2026-06-11 00:00:00",
                        "close": 223.99,
                    },
                ]
            )
        },
    )

    market_update.sync_provider_raw_frames(frames, paths=paths)

    updated_text = existing_path.read_text(encoding="utf-8")
    assert updated_text.startswith(existing_text)
    assert updated_text.count("2026-06-10 00:00:00") == 1
    assert "2026-06-11 00:00:00" in updated_text


def test_update_market_data_module_cli_dispatches(monkeypatch, capsys):
    from src.data import market_update

    seen = {}

    def fake_update_market_data(**kwargs):
        seen.update(kwargs)
        return {"operation": "market_data_update", "provider": kwargs["provider"]}

    monkeypatch.setattr(market_update, "update_market_data", fake_update_market_data)

    assert (
        market_update.main(
            [
                "--provider",
                "futu",
                "--dry-run",
                "--skip-qlib",
                "--skip-registry",
                "--start",
                "2026-06-20",
                "--end",
                "2026-06-24",
            ]
        )
        == 0
    )

    assert seen["provider"] == "futu"
    assert seen["dry_run"] is True
    assert seen["skip_qlib"] is True
    assert seen["skip_registry"] is True
    assert seen["start"] == "2026-06-20"
    assert seen["end"] == "2026-06-24"
    assert '"provider": "futu"' in capsys.readouterr().out
