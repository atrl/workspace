from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_registry_includes_intraday_parquet_metadata(monkeypatch, tmp_path: Path):
    from scripts import update_data_registry

    data = tmp_path / "data"
    intraday_dir = data / "processed" / "intraday"
    intraday_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        [
            {
                "provider": "futu",
                "symbol": "SOXL",
                "interval": "1m",
                "timestamp": pd.Timestamp("2026-06-04 09:30", tz="America/New_York"),
                "date": "2026-06-04",
                "session": "regular",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
                "turnover": 1.0,
                "source_tz": "America/New_York",
                "adjusted": True,
                "fetched_at": "2026-06-05T00:00:00Z",
            }
        ]
    )
    frame.to_parquet(intraday_dir / "soxl_soxx_1m.parquet", index=False)
    for subdir in [
        data / "raw" / "symbols",
        data / "raw" / "external",
        data / "processed" / "panels",
        data / "qlib",
        data / "cache" / "training",
    ]:
        subdir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(update_data_registry, "ROOT", tmp_path)
    monkeypatch.setattr(update_data_registry, "DATA", data)

    registry = update_data_registry.build_registry()

    meta = registry["intraday"]["soxl_soxx_1m"]
    assert meta["path"] == "data/processed/intraday/soxl_soxx_1m.parquet"
    assert meta["rows"] == 1
    assert meta["symbols"] == ["SOXL"]
    assert meta["providers"] == ["futu"]
    assert meta["intervals"] == ["1m"]
    assert meta["start"] == "2026-06-04T09:30:00-04:00"
    assert meta["end"] == "2026-06-04T09:30:00-04:00"
