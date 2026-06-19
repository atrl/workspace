from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data import (
    build_data_update_manifest,
    canonical_snapshot_from_panel,
    inspect_qlib_store,
    load_qlib_calendar,
    write_canonical_snapshot_update,
)


def _make_qlib_store(root: Path) -> Path:
    qlib_dir = root / "qlib" / "soxl"
    (qlib_dir / "calendars").mkdir(parents=True)
    (qlib_dir / "instruments").mkdir()
    (qlib_dir / "features" / "SOXL").mkdir(parents=True)
    (qlib_dir / "features" / "SOXX").mkdir(parents=True)
    (qlib_dir / "calendars" / "day.txt").write_text("2026-01-02\n2026-01-05\n2026-01-06\n", encoding="utf-8")
    (qlib_dir / "instruments" / "all.txt").write_text(
        "SOXL\t2026-01-02\t2026-01-06\nSOXX\t2026-01-02\t2026-01-06\n",
        encoding="utf-8",
    )
    for instrument in ["SOXL", "SOXX"]:
        for field in ["open", "close", "volume"]:
            (qlib_dir / "features" / instrument / f"{field}.day.bin").write_bytes(b"qlib-bin-placeholder")
    return qlib_dir


def test_inspect_qlib_store_reports_calendar_instruments_and_feature_gaps(tmp_path):
    qlib_dir = _make_qlib_store(tmp_path)

    audit = inspect_qlib_store(
        qlib_dir,
        required_instruments=["SOXL", "SOXX"],
        required_fields=["open", "close", "high", "volume"],
    )

    assert audit.provider_uri == qlib_dir.as_posix()
    assert audit.calendar_start == "2026-01-02"
    assert audit.calendar_end == "2026-01-06"
    assert audit.instruments == {
        "SOXL": {"start": "2026-01-02", "end": "2026-01-06"},
        "SOXX": {"start": "2026-01-02", "end": "2026-01-06"},
    }
    assert audit.missing_features == {
        "SOXL": ["high"],
        "SOXX": ["high"],
    }
    assert audit.to_manifest()["status"] == "missing_features"


def test_canonical_snapshot_update_writes_controlled_sidecars_and_blocks_future_rows(tmp_path):
    qlib_dir = _make_qlib_store(tmp_path)
    calendar = load_qlib_calendar(qlib_dir)
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]),
            "close": [10.0, 11.0, 12.0, 13.0],
            "volume": [100, 110, 120, 130],
        }
    )
    snapshot = canonical_snapshot_from_panel(
        panel,
        calendar=calendar,
        instrument="SOXL",
        as_of="2026-01-06",
        source_path=tmp_path / "panel.csv",
    )
    update = write_canonical_snapshot_update(
        snapshot,
        output_dir=tmp_path / "data_updates" / "soxl_20260106",
        qlib_dir=qlib_dir,
    )
    manifest = build_data_update_manifest(update)
    written_manifest = json.loads(update.manifest_path.read_text(encoding="utf-8"))
    written_snapshot = pd.read_csv(update.snapshot_path)

    assert update.snapshot_path.name == "SOXL_2026-01-06_snapshot.csv"
    assert written_snapshot["date"].tolist() == ["2026-01-02", "2026-01-05", "2026-01-06"]
    assert manifest == written_manifest
    assert manifest["instrument"] == "SOXL"
    assert manifest["as_of"] == "2026-01-06"
    assert manifest["row_count"] == 3
    assert manifest["qlib_store"]["calendar_end"] == "2026-01-06"
    assert manifest["snapshot_hash"] == update.snapshot_hash
