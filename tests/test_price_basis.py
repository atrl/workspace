from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.data.price_basis import (
    PriceBasisError,
    build_qualified_price_store,
    load_qlib_price_calendar,
    normalize_futu_candles,
    validate_price_basis,
)
from src.data.qlib_native import build_qlib_store_from_frame


@pytest.fixture
def source_fixture(tmp_path):
    # Deliberately includes a year boundary: missing warmup != model coverage.
    dates = ["2010-12-30", "2010-12-31", "2011-01-03", "2011-01-04"]
    raw = pd.DataFrame([
        {"code": f"US.{symbol}", "time_key": f"{date} 00:00:00",
         "open": 10.123456789 + i, "high": 11.2 + i, "low": 9.0 + i,
         "close": 10.75 + i, "volume": 10000001 + i}
        for symbol in ("SOXL", "SOXX") for i, date in enumerate(dates)
    ])
    reference = raw.assign(instrument=raw.code.str.replace("US.", "", regex=False), datetime=pd.to_datetime(raw.time_key))
    provider = tmp_path / "reference"
    build_qlib_store_from_frame(reference, provider_uri=provider, as_of=dates[-1],
                                field_map={x: x for x in ("open", "high", "low", "close", "volume")})
    calendar = load_qlib_price_calendar(provider, dates[0], dates[-1])
    identity = {
        "source": "Futu", "adjustment": "QFQ", "session": "RTH", "extended_time": False,
        "timezone": "America/New_York", "volume_unit": "shares", "volume_basis": "split_adjusted_shares",
        "retrieval_id": "synthetic-test-source", "requested_start": dates[0], "requested_end": dates[-1],
        "retrieved_at_utc": "2011-01-05T12:00:00Z",
        "raw_artifacts": [{"name": "synthetic-response", "sha256": hashlib.sha256(raw.to_json().encode()).hexdigest()}],
        "rehabilitation": {symbol: {"before_sha256": "a" * 64, "after_sha256": "a" * 64,
                                    "before_observed_at_utc": "2011-01-05T11:00:00Z",
                                    "after_observed_at_utc": "2011-01-05T13:00:00Z"} for symbol in ("SOXL", "SOXX")},
    }
    return raw, identity, calendar


def _audit(raw, identity, calendar):
    frame = normalize_futu_candles(raw, identity)
    return frame, validate_price_basis(frame, calendar, "2011-01-04")


def _manifest(frame, audit, calendar):
    return {"calendar": calendar, "qualification": audit, "source_identities": frame.attrs["source_identities"]}


def test_native_snapshot_content_and_immutable_destination(source_fixture, tmp_path):
    raw, identity, calendar = source_fixture
    frame, audit = _audit(raw, identity, calendar)
    assert audit["qualification_pass"] is True
    assert len(frame) == len(raw)
    assert frame.open.tolist() == raw.open.tolist()
    assert frame.source_raw_row_sha256.str.len().eq(64).all()
    destination = tmp_path / "qualified"
    result = build_qualified_price_store(frame, destination, "2011-01-04", _manifest(frame, audit, calendar))
    assert result["native_readback"]["status"] == "passed"
    assert result["native_readback"]["rows"] == 8
    assert result["native_readback"]["max_absolute_quantization_by_field"]["open"] > 0
    assert json.loads((destination / "price_basis_status.json").read_text())["status"] == "qualified"
    before = (destination / "features/soxl/open.day.bin").read_bytes()
    with pytest.raises(PriceBasisError, match="existing providers are immutable"):
        build_qualified_price_store(frame, destination, "2011-01-04", _manifest(frame, audit, calendar))
    assert (destination / "features/soxl/open.day.bin").read_bytes() == before


def test_reports_all_warmup_and_model_gaps_without_shortening(source_fixture, tmp_path):
    raw, identity, calendar = source_fixture
    frame, audit = _audit(raw.drop(index=[0, 2, 5]), identity, calendar)
    assert audit["missing_dates_full"] == {"SOXL": ["2010-12-30", "2011-01-03"], "SOXX": ["2010-12-31"]}
    assert audit["missing_dates_model"] == {"SOXL": ["2011-01-03"], "SOXX": []}
    assert not audit["qualification_pass"]
    with pytest.raises(PriceBasisError, match="incomplete history cannot be shortened"):
        build_qualified_price_store(frame, tmp_path / "blocked", "2011-01-04", _manifest(frame, audit, calendar))
    assert not (tmp_path / "blocked").exists()
    _, warmup_only = _audit(raw.drop(index=[0]), identity, calendar)
    assert warmup_only["model_coverage_pass"]
    assert not warmup_only["qualification_pass"]


def test_rejects_duplicates_invalid_prices_volume_and_future_without_dropping(source_fixture):
    raw, identity, calendar = source_fixture
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    raw.loc[1, "high"] = 1.0
    raw.loc[2, "volume"] = 0
    raw.loc[3, "low"] = np.inf
    raw["close"] = raw["close"].astype(object)
    raw.loc[4, "close"] = "not-a-price"
    raw.loc[5, "time_key"] = "2011-01-05 00:00:00"
    raw.loc[6, "code"] = "US.UNKNOWN"
    raw.loc[7, "time_key"] = "2011-01-04 16:00:00"
    frame, audit = _audit(raw, identity, calendar)
    assert len(frame) == 9
    assert len(audit["invalid_rows"]) == 9
    issues = {x for row in audit["invalid_rows"] for x in row["issues"]}
    assert {"duplicate_instrument_date", "physical_ohlc_order", "nonfinite_or_nonpositive_ohlcv",
            "after_as_of", "unknown_instrument", "invalid_or_outside_calendar_date"} <= issues
    assert not audit["qualification_pass"]


@pytest.mark.parametrize("change", ["basis", "rehab", "raw", "identity", "row", "retrieval_time", "rehab_time"])
def test_source_basis_and_lineage_are_gates(source_fixture, change):
    raw, identity, calendar = source_fixture
    identity = copy.deepcopy(identity)
    if change == "basis":
        identity["volume_basis"] = "unadjusted_shares"
    elif change == "rehab":
        identity["rehabilitation"]["SOXL"]["after_sha256"] = "b" * 64
    elif change == "raw":
        identity["raw_artifacts"] = []
    elif change == "retrieval_time":
        identity["retrieved_at_utc"] = "2011-01-05 12:00:00"
    elif change == "rehab_time":
        identity["rehabilitation"]["SOXL"]["after_observed_at_utc"] = "2011-01-04T00:00:00Z"
    frame = normalize_futu_candles(raw, identity)
    if change == "identity":
        frame.loc[0, "source_identity_sha256"] = "b" * 64
    elif change == "row":
        frame.loc[0, "open"] += 0.001
    audit = validate_price_basis(frame, calendar, "2011-01-04")
    assert not audit["qualification_pass"]
    assert audit["source_identity_errors"] or audit["invalid_rows"]


def test_stale_or_forged_pass_cannot_build(source_fixture, tmp_path):
    raw, identity, calendar = source_fixture
    frame, audit = _audit(raw, identity, calendar)
    manifest = _manifest(frame, audit, calendar)
    frame.loc[0, "close"] += .01
    with pytest.raises(PriceBasisError, match="no longer matches"):
        build_qualified_price_store(frame, tmp_path / "stale", "2011-01-04", manifest)
    assert not (tmp_path / "stale").exists()
    forged = copy.deepcopy(calendar)
    forged["dates"] = forged["dates"][1:]
    with pytest.raises(PriceBasisError, match="calendar content hash"):
        validate_price_basis(frame, forged, "2011-01-04")


def test_native_readback_corruption_fails_and_never_marks_qualified(source_fixture, tmp_path, monkeypatch):
    import src.data.price_basis as module

    raw, identity, calendar = source_fixture
    frame, audit = _audit(raw, identity, calendar)
    original = module.load_qlib_feature_frame

    def corrupt(*args, **kwargs):
        loaded = original(*args, **kwargs)
        loaded.iloc[0, 0] += 1
        return loaded

    monkeypatch.setattr(module, "load_qlib_feature_frame", corrupt)
    destination = tmp_path / "failed-readback"
    with pytest.raises(PriceBasisError, match="D.features readback differs"):
        build_qualified_price_store(frame, destination, "2011-01-04", _manifest(frame, audit, calendar))
    assert json.loads((destination / "price_basis_status.json").read_text())["status"] == "failed"
    assert not (destination / "price_basis_manifest.json").exists()


def test_float32_nonrepresentable_values_block_before_native_build(source_fixture):
    raw, identity, calendar = source_fixture
    raw["volume"] = raw["volume"].astype(float)
    raw.loc[0, "volume"] = 1e100
    _, audit = _audit(raw, identity, calendar)
    assert not audit["qualification_pass"]
    assert "not_positive_finite_in_qlib_float32" in audit["invalid_rows"][0]["issues"]
