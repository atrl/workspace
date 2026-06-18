from __future__ import annotations

import json

import pandas as pd

import src.data as data_api


def test_build_canonical_snapshot_aligns_to_calendar_and_records_schema_hash(tmp_path):
    raw = pd.DataFrame(
        [
            {"date": "2026-01-02", "instrument": "SOXL", "close": 10.0, "volume": 100},
            {"date": "2026-01-03", "instrument": "SOXL", "close": 11.0, "volume": 110},
            {"date": "2026-01-05", "instrument": "SOXL", "close": 12.0, "volume": 120},
        ]
    )
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-01-02", "2026-01-05"]), name="date")

    snapshot = data_api.build_canonical_snapshot(
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

    result = data_api.write_canonical_snapshot(snapshot, tmp_path)

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
        data_api.build_canonical_snapshot(
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


def test_publish_canonical_snapshot_writes_manifest_and_updates_registry(tmp_path):
    raw = pd.DataFrame(
        [
            {"date": "2026-01-02", "instrument": "SOXL", "close": 10.0, "volume": 100},
            {"date": "2026-01-05", "instrument": "SOXL", "close": 12.0, "volume": 120},
        ]
    )
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-01-02", "2026-01-05"]), name="date")
    snapshot = data_api.build_canonical_snapshot(
        raw,
        name="unit_soxl_daily",
        universe=("SOXL",),
        calendar=calendar,
        calendar_path="data/qlib/soxl/calendars/day.txt",
        source_paths=("data/raw/unit.csv",),
        start="2026-01-01",
        end="2026-01-05",
        as_of="2026-01-05",
    )

    result = data_api.publish_canonical_snapshot(
        snapshot,
        output_dir=tmp_path / "data" / "snapshots" / "unit_soxl_daily" / "2026-01-05",
        dataset_key="daily.unit_soxl",
        registry_path=tmp_path / "data" / "data_registry.json",
        published_at="2026-01-05T21:00:00Z",
    )

    assert result["schema_version"] == "data.update_manifest.v1"
    assert result["dataset_key"] == "daily.unit_soxl"
    assert result["status"] == "published"
    assert result["paths"]["frame"] == "canonical_frame.csv"
    assert result["paths"]["dataset_snapshot"] == "dataset_snapshot.json"
    assert result["paths"]["manifest"] == "data_update_manifest.json"
    assert result["published_at"] == "2026-01-05T21:00:00Z"
    assert result["dataset"]["metadata"]["row_count"] == 2
    assert len(result["dataset_schema_hash"]) == 64
    assert len(result["manifest_hash"]) == 64

    manifest = json.loads((tmp_path / "data" / "snapshots" / "unit_soxl_daily" / "2026-01-05" / "data_update_manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((tmp_path / "data" / "data_registry.json").read_text(encoding="utf-8"))

    assert manifest == result
    assert registry["schema_version"] == "data.registry.v1"
    assert registry["datasets"]["daily.unit_soxl"]["current"]["manifest_hash"] == result["manifest_hash"]
    assert registry["datasets"]["daily.unit_soxl"]["current"]["snapshot_path"].endswith(
        "data/snapshots/unit_soxl_daily/2026-01-05"
    )
    assert registry["datasets"]["daily.unit_soxl"]["history"][0]["manifest_hash"] == result["manifest_hash"]


def test_publish_canonical_snapshot_registry_preserves_previous_history(tmp_path):
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]), name="date")
    first = data_api.build_canonical_snapshot(
        pd.DataFrame(
            [
                {"date": "2026-01-02", "instrument": "SOXL", "close": 10.0},
                {"date": "2026-01-05", "instrument": "SOXL", "close": 12.0},
            ]
        ),
        name="unit_soxl_daily",
        universe=("SOXL",),
        calendar=calendar,
        calendar_path="data/qlib/soxl/calendars/day.txt",
        source_paths=("data/raw/unit.csv",),
        start="2026-01-01",
        end="2026-01-05",
        as_of="2026-01-05",
    )
    second = data_api.build_canonical_snapshot(
        pd.DataFrame(
            [
                {"date": "2026-01-02", "instrument": "SOXL", "close": 10.0},
                {"date": "2026-01-05", "instrument": "SOXL", "close": 12.0},
                {"date": "2026-01-06", "instrument": "SOXL", "close": 13.0},
            ]
        ),
        name="unit_soxl_daily",
        universe=("SOXL",),
        calendar=calendar,
        calendar_path="data/qlib/soxl/calendars/day.txt",
        source_paths=("data/raw/unit.csv",),
        start="2026-01-01",
        end="2026-01-06",
        as_of="2026-01-06",
    )

    first_result = data_api.publish_canonical_snapshot(
        first,
        output_dir=tmp_path / "data" / "snapshots" / "unit_soxl_daily" / "2026-01-05",
        dataset_key="daily.unit_soxl",
        registry_path=tmp_path / "data" / "data_registry.json",
        published_at="2026-01-05T21:00:00Z",
    )
    second_result = data_api.publish_canonical_snapshot(
        second,
        output_dir=tmp_path / "data" / "snapshots" / "unit_soxl_daily" / "2026-01-06",
        dataset_key="daily.unit_soxl",
        registry_path=tmp_path / "data" / "data_registry.json",
        published_at="2026-01-06T21:00:00Z",
    )

    registry = json.loads((tmp_path / "data" / "data_registry.json").read_text(encoding="utf-8"))
    current = registry["datasets"]["daily.unit_soxl"]["current"]
    history = registry["datasets"]["daily.unit_soxl"]["history"]

    assert current["manifest_hash"] == second_result["manifest_hash"]
    assert current["end"] == "2026-01-06"
    assert [item["manifest_hash"] for item in history] == [
        first_result["manifest_hash"],
        second_result["manifest_hash"],
    ]
