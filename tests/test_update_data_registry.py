from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts import update_data_registry


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_minimal_registry_surface(root: Path) -> None:
    for symbol in ["SOXL", "SOXX", "TQQQ", "UPRO", "SPXL", "TECL"]:
        _write_csv(
            root / "data/raw/symbols" / f"{symbol}.csv",
            [{"date": "2024-01-02", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 100}],
        )
    _write_csv(
        root / "data/processed/panels/soxl_soxx_vix_full.csv",
        [{"date": "2024-01-02", "soxl_close": 2, "soxx_close": 3, "vix": 20}],
    )
    qlib = root / "data/qlib/SOXL"
    (qlib / "calendars").mkdir(parents=True, exist_ok=True)
    (qlib / "instruments").mkdir(parents=True, exist_ok=True)
    (qlib / "calendars/day.txt").write_text("2024-01-02\n", encoding="utf-8")
    (qlib / "instruments/all.txt").write_text("SOXL\t2024-01-02\t2024-01-02\n", encoding="utf-8")
    (root / "data/raw/external").mkdir(parents=True, exist_ok=True)
    (root / "data/cache/training").mkdir(parents=True, exist_ok=True)


def _patch_registry_root(root: Path):
    original = (update_data_registry.ROOT, update_data_registry.DATA, update_data_registry.REGISTRY_PATH)
    update_data_registry.ROOT = root
    update_data_registry.DATA = root / "data"
    update_data_registry.REGISTRY_PATH = root / "data/datasets.yaml"
    return original


def _restore_registry_root(original) -> None:
    update_data_registry.ROOT, update_data_registry.DATA, update_data_registry.REGISTRY_PATH = original


def _intraday_rows(extra: bool = False) -> pd.DataFrame:
    rows = [
        {
            "provider": "polygon",
            "symbol": "SOXL",
            "interval": "1m",
            "timestamp": pd.Timestamp("2024-01-02 09:30:00", tz="America/New_York"),
            "date": "2024-01-02",
            "session": "regular",
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.1,
            "volume": 100.0,
            "turnover": 0.0,
            "source_tz": "America/New_York",
            "adjusted": True,
            "fetched_at": pd.Timestamp("2026-06-05T00:00:00Z"),
        },
        {
            "provider": "polygon",
            "symbol": "SOXX",
            "interval": "1m",
            "timestamp": pd.Timestamp("2024-01-02 09:30:00", tz="America/New_York"),
            "date": "2024-01-02",
            "session": "regular",
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.4,
            "volume": 200.0,
            "turnover": 0.0,
            "source_tz": "America/New_York",
            "adjusted": True,
            "fetched_at": pd.Timestamp("2026-06-05T00:00:00Z"),
        },
    ]
    if extra:
        rows.append({**rows[-1], "timestamp": pd.Timestamp("2024-01-02 09:31:00", tz="America/New_York")})
    return pd.DataFrame(rows)


def test_data_registry_builds_intraday_parquet_metadata():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_minimal_registry_surface(root)
        raw_path = root / "data/raw/intraday/polygon/SOXL_1m_2024-01-02_2024-01-02_regular.parquet"
        processed_path = root / "data/processed/intraday/soxl_soxx_1m.parquet"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        _intraday_rows().iloc[:1].to_parquet(raw_path, index=False)
        _intraday_rows().to_parquet(processed_path, index=False)

        original = _patch_registry_root(root)
        try:
            registry = update_data_registry.build_registry()
        finally:
            _restore_registry_root(original)

        intraday = registry["intraday"]
        raw_meta = intraday["raw"]["polygon"]["SOXL_1m_2024-01-02_2024-01-02_regular"]
        processed_meta = intraday["processed"]["soxl_soxx_1m"]
        assert raw_meta["rows"] == 1
        assert raw_meta["symbols"] == ["SOXL"]
        assert raw_meta["intervals"] == ["1m"]
        assert raw_meta["sessions"] == ["regular"]
        assert processed_meta["rows"] == 2
        assert processed_meta["min_timestamp"] == "2024-01-02T09:30:00-05:00"
        assert processed_meta["max_timestamp"] == "2024-01-02T09:30:00-05:00"


def test_data_registry_check_catches_stale_intraday_rows_and_timestamps():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_minimal_registry_surface(root)
        processed_path = root / "data/processed/intraday/soxl_soxx_1m.parquet"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        _intraday_rows().to_parquet(processed_path, index=False)

        original = _patch_registry_root(root)
        try:
            registry = update_data_registry.build_registry()
            _intraday_rows(extra=True).to_parquet(processed_path, index=False)
            current = update_data_registry.build_registry()
            failures = update_data_registry.check_registry(registry, current_registry=current)
        finally:
            _restore_registry_root(original)

        assert any("stale registry intraday.processed.soxl_soxx_1m.rows" in failure for failure in failures)
        assert any("stale registry intraday.processed.soxl_soxx_1m.max_timestamp" in failure for failure in failures)
