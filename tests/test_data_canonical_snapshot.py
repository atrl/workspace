from __future__ import annotations

import json

import pandas as pd

from src.data import build_canonical_snapshot, write_canonical_snapshot


def test_build_canonical_snapshot_aligns_to_calendar_and_records_schema_hash(tmp_path):
    raw = pd.DataFrame(
        [
            {"date": "2026-01-02", "instrument": "SOXL", "close": 10.0, "volume": 100},
            {"date": "2026-01-03", "instrument": "SOXL", "close": 11.0, "volume": 110},
            {"date": "2026-01-05", "instrument": "SOXL", "close": 12.0, "volume": 120},
        ]
    )
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-01-02", "2026-01-05"]), name="date")

    snapshot = build_canonical_snapshot(
        raw,
        name="unit_soxl_daily",
        universe=("SOXL",),
        calendar=calendar,
        calendar_path="data/qlib/soxl/calendars/day.txt",
        source_paths=("raw/unit.csv",),
        start="2026-01-01",
        end="2026-01-05",
        as_of="2026-01-05",
    )

    assert snapshot.frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-05"]
    assert snapshot.dataset.name == "unit_soxl_daily"
    assert snapshot.dataset.start == "2026-01-02"
    assert snapshot.dataset.end == "2026-01-05"
    assert snapshot.dataset.data_paths == ("raw/unit.csv",)
    assert snapshot.dataset.metadata["row_count"] == 2
    assert snapshot.dataset.metadata["dropped_non_calendar_rows"] == 1
    assert snapshot.dataset.metadata["as_of"] == "2026-01-05"
    assert len(snapshot.dataset.metadata["frame_schema_hash"]) == 64
    assert len(snapshot.dataset.schema_hash()) == 64

    result = write_canonical_snapshot(snapshot, tmp_path)

    assert result["schema_version"] == "data.canonical_snapshot.export.v1"
    assert result["paths"]["frame"] == "canonical_frame.csv"
    assert result["paths"]["dataset_snapshot"] == "dataset_snapshot.json"
    saved_frame = pd.read_csv(tmp_path / "canonical_frame.csv")
    assert saved_frame["date"].tolist() == ["2026-01-02", "2026-01-05"]
    saved_snapshot = json.loads((tmp_path / "dataset_snapshot.json").read_text(encoding="utf-8"))
    assert saved_snapshot["dataset"]["metadata"]["row_count"] == 2
    assert saved_snapshot["dataset_schema_hash"] == snapshot.dataset.schema_hash()


def test_build_canonical_snapshot_rejects_future_rows_before_calendar_filter():
    raw = pd.DataFrame(
        [
            {"date": "2026-01-02", "instrument": "SOXL", "close": 10.0},
            {"date": "2026-01-06", "instrument": "SOXL", "close": 12.0},
        ]
    )
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-01-02", "2026-01-06"]), name="date")

    try:
        build_canonical_snapshot(
            raw,
            name="unit_soxl_daily",
            universe=("SOXL",),
            calendar=calendar,
            calendar_path="data/qlib/soxl/calendars/day.txt",
            source_paths=("raw/unit.csv",),
            start="2026-01-01",
            end="2026-01-06",
            as_of="2026-01-05",
        )
    except ValueError as exc:
        assert "future rows after as_of" in str(exc)
    else:
        raise AssertionError("expected ValueError")
