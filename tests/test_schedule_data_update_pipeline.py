from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.schedule import DataUpdatePipelineConfig, classify_market_calendar_day, run_data_update_once


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2026-01-02", "instrument": "SOXL", "close": 10.0, "volume": 100},
            {"date": "2026-01-03", "instrument": "SOXL", "close": 11.0, "volume": 110},
            {"date": "2026-01-05", "instrument": "SOXL", "close": 12.0, "volume": 120},
        ]
    )


def _calendar() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(["2026-01-02", "2026-01-05"]), name="date")


def _config(tmp_path: Path) -> DataUpdatePipelineConfig:
    return DataUpdatePipelineConfig(
        name="daily_soxl_data_update",
        dataset_key="daily.soxl_unit",
        dataset_name="unit_soxl_daily",
        universe=("SOXL",),
        output_dir=tmp_path / "data" / "snapshots" / "unit_soxl_daily" / "2026-01-05",
        registry_path=tmp_path / "data" / "data_registry.json",
        calendar_path="data/qlib/soxl/calendars/day.txt",
        source_paths=("provider:unit",),
        start="2026-01-01",
        end="2026-01-05",
        as_of="2026-01-05",
        published_at="2026-01-05T21:00:00Z",
        notify_details=True,
        retries=1,
    )


def test_data_update_pipeline_publishes_canonical_snapshot_with_lock_registry_and_notify(tmp_path):
    events: list[str] = []
    config = _config(tmp_path)

    result = run_data_update_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        frame_provider=lambda _: _raw_frame(),
        calendar_provider=lambda _: _calendar(),
    )

    manifest_path = config.output_dir / "data_update_manifest.json"
    registry_path = config.registry_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    assert result.status == "success"
    assert result.payload["schema_version"] == "schedule.data_update_result.v1"
    assert result.payload["dataset_key"] == "daily.soxl_unit"
    assert result.payload["row_count"] == 2
    assert result.payload["dropped_non_calendar_rows"] == 1
    assert result.payload["as_of"] == "2026-01-05"
    assert result.payload["manifest_path"].endswith("data_update_manifest.json")
    assert result.payload["manifest_hash"] == manifest["manifest_hash"]
    assert result.payload["registry_path"].endswith("data_registry.json")
    assert result.payload["attempts"] == 1
    assert len(result.payload["frame_schema_hash"]) == 64
    assert registry["datasets"]["daily.soxl_unit"]["current"]["manifest_hash"] == manifest["manifest_hash"]
    assert events == [
        "daily_soxl_data_update:success | dataset=daily.soxl_unit | rows=2 | as_of=2026-01-05 "
        f"| attempts=1 | manifest={manifest_path}"
    ]
    assert not (tmp_path / "locks" / "daily_soxl_data_update.lock").exists()


def test_data_update_pipeline_retries_transient_provider_failure_before_success(tmp_path):
    config = _config(tmp_path)
    calls = {"count": 0}

    def flaky_frame_provider(_: DataUpdatePipelineConfig) -> pd.DataFrame:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary provider failure")
        return _raw_frame()

    result = run_data_update_once(
        config,
        lock_dir=tmp_path / "locks",
        frame_provider=flaky_frame_provider,
        calendar_provider=lambda _: _calendar(),
    )

    assert result.status == "success"
    assert result.payload["attempts"] == 2
    assert calls["count"] == 2


def test_data_update_pipeline_skips_closed_market_day_before_provider_lock_or_notify(tmp_path):
    events: list[str] = []
    calls = {"frame": 0, "calendar": 0}
    decision = classify_market_calendar_day(
        "2026-01-01",
        trading_calendar=("2026-01-02",),
        holiday_reasons={"2026-01-01": "New Year's Day"},
    )
    config = DataUpdatePipelineConfig(
        **{**_config(tmp_path).__dict__, "market_calendar_decision": decision}
    )

    def frame_provider(_: DataUpdatePipelineConfig) -> pd.DataFrame:
        calls["frame"] += 1
        return _raw_frame()

    def calendar_provider(_: DataUpdatePipelineConfig) -> pd.DatetimeIndex:
        calls["calendar"] += 1
        return _calendar()

    result = run_data_update_once(
        config,
        lock_dir=tmp_path / "locks",
        notify=events.append,
        frame_provider=frame_provider,
        calendar_provider=calendar_provider,
    )

    assert result.status == "skipped"
    assert result.detail == "New Year's Day"
    assert result.payload["market_calendar_decision"]["market_status"] == "closed"
    assert calls == {"frame": 0, "calendar": 0}
    assert events == []
    assert not (tmp_path / "locks" / "daily_soxl_data_update.lock").exists()
    assert not config.output_dir.exists()
