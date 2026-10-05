"""VIX source completeness, immutable provenance, and actual Qlib readback."""
import json

import numpy as np
import pandas as pd
import pytest

from src.data import legacy_vix_source as source
from src.data.qlib_native import build_qlib_store_from_frame


def _raw(dates, closes):
    stamps = (pd.DatetimeIndex(dates).tz_localize("America/Chicago") + pd.Timedelta(hours=16)).as_unit("s").asi8.tolist()
    return json.dumps({"chart": {"error": None, "result": [{
        "meta": {"symbol": "^VIX", "instrumentType": "INDEX", "dataGranularity": "1d", "exchangeTimezoneName": "America/Chicago"},
        "timestamp": stamps, "indicators": {"quote": [{"close": closes}]},
    }]}}).encode()


def test_holiday_projection_does_not_fill_missing_session():
    calendar = pd.DatetimeIndex(["2024-07-03", "2024-07-05"], name="datetime")
    frame, audit = source.parse_yahoo_vix_close(_raw(["2024-07-03", "2024-07-04", "2024-07-05"], [12., None, 13.]), calendar)
    assert frame["close"].tolist() == [12., 13.]
    assert audit["excluded_noncanonical_dates"] == ["2024-07-04"]
    assert audit["excluded_noncanonical_null_closes"] == 1
    with pytest.raises(ValueError, match="no filling permitted"):
        source.parse_yahoo_vix_close(_raw(["2024-07-03"], [12.]), calendar)


@pytest.mark.parametrize("closes", [[12., None], [12., 0.], [12., -1.], [12., float("inf")]])
def test_invalid_canonical_close_fails_closed(closes):
    calendar = pd.date_range("2024-07-02", periods=2)
    with pytest.raises(ValueError, match="no filling permitted"):
        source.parse_yahoo_vix_close(_raw(calendar, closes), calendar)


def test_future_observations_and_duplicate_days_fail_closed():
    calendar = pd.date_range("2024-07-02", periods=2)
    with pytest.raises(ValueError, match="exceed frozen"):
        source.parse_yahoo_vix_close(_raw(pd.date_range("2024-07-02", periods=3), [12., 13., 14.]), calendar)
    with pytest.raises(ValueError, match="duplicate or unordered"):
        source.parse_yahoo_vix_close(_raw(["2024-07-02", "2024-07-02"], [12., 13.]), calendar)


def test_native_roundtrip_immutable_hash_and_overwrite_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "START", "2024-01-02")
    monkeypatch.setattr(source, "AS_OF", "2024-01-04")
    calendar = pd.date_range(source.START, source.AS_OF)
    canonical = pd.DataFrame({"instrument": "SOXL", "datetime": calendar, "close": [10., 11., 12.]})
    build_qlib_store_from_frame(canonical, provider_uri=tmp_path / source.CANONICAL_PROVIDER,
                               as_of=source.AS_OF, field_map={"close": "close"})
    raw_path = tmp_path / "source.json"
    raw_path.write_bytes(_raw(calendar, [12.3, 13.4, 14.5]))
    manifest = source.freeze_legacy_vix_snapshot(tmp_path, raw_path, source_url="synthetic test fixture",
                                               retrieved_at="2024-01-05T00:00:00+00:00")
    actual, audit = source.load_legacy_vix(tmp_path)
    np.testing.assert_array_equal(actual.values, np.array([12.3, 13.4, 14.5], dtype=np.float32))
    assert audit["native_readback"]["status"] == "passed"
    assert audit["availability"]["historical_publication_vintage_verified"] is False
    assert manifest["native_readback"]["source_api"][-1] == "qlib.data.D.features"
    with pytest.raises(ValueError, match="immutable"):
        source.freeze_legacy_vix_snapshot(tmp_path, raw_path, source_url="fixture", retrieved_at="fixture")
    feature = tmp_path / source.VIX_PROVIDER / "features/vix/close.day.bin"
    feature.write_bytes(feature.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash changed"):
        source.load_legacy_vix(tmp_path)
