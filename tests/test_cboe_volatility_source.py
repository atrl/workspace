"""Full raw intake and lag-only features tested with real temporary Qlib stores."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from src.data.cboe_volatility_source import FEATURES, SYMBOLS, intake_cboe_source, load_cboe_features


@pytest.fixture
def raw_inputs(tmp_path):
    dates = pd.DatetimeIndex(["2014-01-30", "2014-01-31", "2014-02-03", "2014-02-04",
                              "2014-02-05", "2014-02-06", "2014-02-07", "2026-10-08"])
    files, metadata = {}, {}
    for symbol, offset in (("VIX9D", 10), ("VIX3M", 20), ("VVIX", 80)):
        use = dates[1:] if symbol == "VVIX" else dates
        values = np.arange(len(dates), dtype=float) + offset
        values = values[1:] if symbol == "VVIX" else values
        columns = {"DATE": use.strftime("%m/%d/%Y")}
        if symbol == "VVIX":
            columns["VVIX"] = values
        else:
            columns.update(OPEN=values, HIGH=values + 1., LOW=values - 1., CLOSE=values)
        path = tmp_path / (symbol + "_official.csv")
        pd.DataFrame(columns).to_csv(path, index=False)
        files[symbol] = path
        metadata[symbol] = {"requested_url": f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv",
            "final_url": f"https://cdn-api.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv",
            "retrieved_at_utc": "2026-10-09T03:25:00+00:00",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    return {"source_files": files, "source_metadata": metadata, "provider_uri": tmp_path / "provider",
        "snapshot_dir": tmp_path / "snapshot", "store_as_of": "2026-10-08"}, dates[:-1]


def rewrite_source(arguments, symbol, transform):
    path = arguments["source_files"][symbol]
    frame = transform(pd.read_csv(path))
    frame.to_csv(path, index=False)
    arguments["source_metadata"][symbol]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    arguments["source_metadata"][symbol]["bytes"] = path.stat().st_size


def test_full_history_native_close_only_store_and_exact_lagged_features(raw_inputs):
    from qlib.data import D

    arguments, calendar = raw_inputs
    originals = {symbol: path.read_bytes() for symbol, path in arguments["source_files"].items()}
    manifest = intake_cboe_source(**arguments)
    assert manifest["store_as_of"] == "2026-10-08"
    assert manifest["raw_rows"] == 23 and manifest["rows_dropped"] == 0
    assert manifest["native_fields"] == ["$close"] and manifest["synthetic_ohlc"] is False
    assert manifest["historical_vintage_certified"] is False
    for symbol in SYMBOLS:
        assert (arguments["snapshot_dir"] / (symbol + ".csv")).read_bytes() == originals[symbol]
        names = [p.name for p in (arguments["provider_uri"] / "features" / symbol.lower()).iterdir()]
        assert names == ["close.day.bin"]
    assert pd.Timestamp(D.calendar()[-1]) == pd.Timestamp("2026-10-08")
    frame, audit = load_cboe_features(arguments["provider_uri"], market_calendar=calendar, market_as_of="2014-02-07")
    assert frame.index.tolist() == list(pd.bdate_range("2014-02-04", "2014-02-07"))
    np.testing.assert_allclose(frame.iloc[0][list(FEATURES)].to_numpy(float), [np.log(.22), np.log(12 / 22), np.log(.82)])
    assert frame.source_date.iloc[0] == pd.Timestamp("2014-02-03")
    assert frame.available_at.iloc[0] == pd.Timestamp("2014-02-03T17:00:00", tz="America/New_York")
    assert frame.decision_at.iloc[0] == pd.Timestamp("2014-02-04T16:15:00", tz="America/New_York")
    assert (frame.available_at < frame.decision_at).all()
    assert audit["store_as_of"] == "2026-10-08" and audit["market_as_of"] == "2014-02-07"
    assert not audit["historical_vintage_certified"] and not audit["availability_timestamp_verified"]
    assert audit["rows_dropped_for_missingness"] == 0
    assert all(path.read_bytes() == originals[symbol] for symbol, path in arguments["source_files"].items())


def test_feature_reads_explicitly_stop_at_market_asof(raw_inputs, monkeypatch):
    import src.data.cboe_volatility_source as source
    from qlib.data import D

    arguments, calendar = raw_inputs
    intake_cboe_source(**arguments)
    real_readback = source._native_readback
    calls = []

    def guarded_readback(*args, **kwargs):
        result = real_readback(*args, **kwargs)
        real_features = D.features
        def guarded_features(*args, **options):
            calls.append(options)
            assert pd.Timestamp(options["end_time"]) == pd.Timestamp("2014-02-07")
            return real_features(*args, **options)
        monkeypatch.setattr(D, "features", guarded_features)
        return result

    monkeypatch.setattr(source, "_native_readback", guarded_readback)
    load_cboe_features(arguments["provider_uri"], market_calendar=calendar, market_as_of="2014-02-07")
    assert len(calls) == 1


def test_identical_intake_is_read_only_and_changed_source_is_rejected(raw_inputs):
    arguments, _ = raw_inputs
    manifest = intake_cboe_source(**arguments)
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for folder in (arguments["snapshot_dir"], arguments["provider_uri"]) for path in folder.rglob("*") if path.is_file()}
    assert intake_cboe_source(**arguments) == manifest
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == value for path, value in hashes.items())
    rewrite_source(arguments, "VIX9D", lambda frame: frame.assign(CLOSE=frame.CLOSE + 1))
    with pytest.raises(ValueError, match="immutable Cboe source identity"):
        intake_cboe_source(**arguments)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == value for path, value in hashes.items())


@pytest.mark.parametrize("corruption", ["schema", "duplicate", "unsorted", "nan", "zero", "negative", "future", "short_endpoint", "storage_overflow"])
def test_invalid_full_source_fails_before_snapshot_or_provider_creation(raw_inputs, corruption):
    arguments, _ = raw_inputs
    def change(frame):
        if corruption == "schema": return frame.rename(columns={"CLOSE": "PRICE"})
        if corruption == "duplicate": return pd.concat([frame, frame.iloc[[-1]]])
        if corruption == "unsorted": return frame.iloc[::-1]
        if corruption == "future": frame.loc[len(frame) - 1, "DATE"] = "10/09/2026"
        elif corruption == "short_endpoint": return frame.iloc[:-1]
        else: frame.loc[0, "CLOSE"] = {"nan": np.nan, "zero": 0, "negative": -1, "storage_overflow": 1e100}[corruption]
        return frame
    rewrite_source(arguments, "VIX9D", change)
    with pytest.raises(ValueError, match="blocked"):
        intake_cboe_source(**arguments)
    assert not arguments["provider_uri"].exists() and not arguments["snapshot_dir"].exists()


def test_future_rows_are_not_silently_clipped_to_market_asof(raw_inputs):
    arguments, _ = raw_inputs
    arguments["store_as_of"] = "2014-02-07"
    with pytest.raises(ValueError, match="no truncation"):
        intake_cboe_source(**arguments)
    assert not arguments["provider_uri"].exists()


@pytest.mark.parametrize("field,value", [("sha256", "different"), ("bytes", 1),
    ("requested_url", "https://unverified.example/VIX9D.csv"), ("retrieved_at_utc", "2026-10-09")])
def test_download_provenance_is_checked_before_any_write(raw_inputs, field, value):
    arguments, _ = raw_inputs
    arguments["source_metadata"]["VIX9D"][field] = value
    with pytest.raises(ValueError, match="blocked"):
        intake_cboe_source(**arguments)
    assert not arguments["provider_uri"].exists() and not arguments["snapshot_dir"].exists()


def test_missing_required_source_session_is_not_filled_or_removed(raw_inputs):
    arguments, calendar = raw_inputs
    rewrite_source(arguments, "VIX3M", lambda frame: frame.loc[frame.DATE != "02/04/2014"])
    intake_cboe_source(**arguments)
    with pytest.raises(ValueError, match="missing or invalid required lagged"):
        load_cboe_features(arguments["provider_uri"], market_calendar=calendar, market_as_of="2014-02-07")


def test_post_market_source_values_do_not_change_any_consumed_feature(raw_inputs, tmp_path):
    arguments, calendar = raw_inputs
    intake_cboe_source(**arguments)
    first, _ = load_cboe_features(arguments["provider_uri"], market_calendar=calendar, market_as_of="2014-02-07")
    for symbol in SYMBOLS:
        def future(frame):
            column = "VVIX" if symbol == "VVIX" else "CLOSE"
            frame.loc[len(frame) - 1, column] *= 10
            return frame
        rewrite_source(arguments, symbol, future)
    arguments["provider_uri"], arguments["snapshot_dir"] = tmp_path / "provider2", tmp_path / "snapshot2"
    intake_cboe_source(**arguments)
    second, _ = load_cboe_features(arguments["provider_uri"], market_calendar=calendar, market_as_of="2014-02-07")
    pd.testing.assert_frame_equal(first, second, check_exact=True)


@pytest.mark.parametrize("target", ["snapshot", "provider"])
def test_existing_unknown_or_tampered_destinations_are_not_overwritten(raw_inputs, target):
    arguments, calendar = raw_inputs
    if target == "snapshot":
        arguments["snapshot_dir"].mkdir()
        retained = arguments["snapshot_dir"] / "VIX9D.csv"
    else:
        arguments["provider_uri"].mkdir()
        retained = arguments["provider_uri"] / "unrelated.txt"
    retained.write_bytes(b"must remain unchanged")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        intake_cboe_source(**arguments)
    assert retained.read_bytes() == b"must remain unchanged"


def test_changed_native_bytes_and_unqualified_calendar_fail_closed(raw_inputs):
    arguments, calendar = raw_inputs
    intake_cboe_source(**arguments)
    with pytest.raises(ValueError, match="benchmark calendar"):
        load_cboe_features(arguments["provider_uri"], market_calendar=calendar[3:], market_as_of="2014-02-07")
    feature = arguments["provider_uri"] / "features/vvix/close.day.bin"
    feature.write_bytes(feature.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="provider bytes changed"):
        load_cboe_features(arguments["provider_uri"], market_calendar=calendar, market_as_of="2014-02-07")
