import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.earnings_surprise import (
    SYMBOLS, FEATURE_COLUMNS, parse_earnings_events, align_earnings_surprise,
    build_earnings_surprise_provider, load_earnings_surprise_features, sha256_file,
)


def payload(symbol, dates=("2024-01-04",), actual="2", estimate="1"):
    return {"symbol": symbol, "quarterlyEarnings": [
        {"reportedDate": d, "fiscalDateEnding": str((pd.Timestamp(d) - pd.Timedelta(days=30)).date()),
         "reportedEPS": actual, "estimatedEPS": estimate, "reportTime": "post-market"} for d in dates]}


def parsed(symbol, **kwargs):
    return parse_earnings_events(payload(symbol, **kwargs), symbol, as_of="2024-12-31",
                                 source_sha256="a" * 64, source_url="https://example.test/earnings")


def test_conservative_weekday_and_weekend_availability():
    events = pd.concat([parsed("NVDA"), parsed("AMD", dates=("2024-01-06",)),
                        parsed("INTC", dates=("2024-01-05",))])
    calendar = pd.bdate_range("2024-01-04", "2024-01-10")
    frame, audit = align_earnings_surprise(events, calendar)
    assert frame.loc["2024-01-04"].isna().all()
    assert frame.loc["2024-01-05", "earnings_surprise_nvda"] == pytest.approx(1 / 3)
    assert pd.isna(frame.loc["2024-01-05", "earnings_surprise_amd"])
    assert frame.loc["2024-01-08", "earnings_surprise_amd"] == pytest.approx(1 / 3)
    assert audit.loc[audit.symbol.eq("AMD"), "earliest_trade_session"].item() == "2024-01-09"
    assert audit.loc[audit.symbol.eq("NVDA"), "earliest_trade_session"].item() == "2024-01-08"


def test_missing_report_invalidates_previous_state():
    parts = [parsed(s, dates=("2023-10-01",)) for s in SYMBOLS]
    parts += [parsed("NVDA", dates=("2024-01-05",), estimate="None"),
              parsed("NVDA", dates=("2024-04-05",), actual="0", estimate="0")]
    frame, _ = align_earnings_surprise(pd.concat(parts), pd.bdate_range("2024-01-04", "2024-04-10"))
    assert frame.loc["2024-01-05", "earnings_surprise_nvda"] == pytest.approx(1 / 3)
    assert pd.isna(frame.loc["2024-01-08", "earnings_surprise_nvda"])
    assert pd.isna(frame.loc["2024-04-05", "earnings_surprise_nvda"])
    assert frame.loc["2024-04-08", "earnings_surprise_nvda"] == 0


def test_future_raw_fails_before_alignment():
    with pytest.raises(ValueError, match="exceeds as_of"):
        parse_earnings_events(payload("NVDA", dates=("2025-01-01",)), "NVDA", as_of="2024-12-31",
                              source_sha256="a" * 64, source_url="https://example.test")


def test_conflicting_duplicates_and_source_identity_fail():
    source = payload("NVDA")
    source["quarterlyEarnings"].append(dict(source["quarterlyEarnings"][0], reportedEPS="3"))
    with pytest.raises(ValueError, match="conflicting"):
        parse_earnings_events(source, "NVDA", as_of="2024-12-31", source_sha256="a" * 64,
                              source_url="https://example.test")
    with pytest.raises(ValueError, match="symbol mismatch"):
        parse_earnings_events(payload("AMD"), "NVDA", as_of="2024-12-31", source_sha256="a" * 64,
                              source_url="https://example.test")


def test_missing_company_and_nonfinite_values_fail():
    with pytest.raises(ValueError, match="three"):
        align_earnings_surprise(parsed("NVDA"), pd.bdate_range("2024-01-01", "2024-01-10"))
    with pytest.raises(ValueError, match="finite"):
        parsed("NVDA", actual="inf")


def test_surprise_is_scale_invariant_and_bounded():
    assert parsed("NVDA", actual="-2", estimate="-1").surprise.item() == pytest.approx(-1 / 3)
    assert parsed("NVDA", actual="20", estimate="10").surprise.item() == pytest.approx(1 / 3)
    assert parsed("NVDA", actual="-1", estimate="1").surprise.item() == -1


@pytest.mark.parametrize("tamper_target", ["feature", "calendar", "raw_source", "event_audit"])
def test_native_provider_readback_and_tamper(tmp_path, tamper_target):
    from src.data.qlib_native import build_qlib_store_from_frame
    repo = tmp_path / "repo"
    reference = repo / "data/qlib/reference"
    snapshot = repo / "data/snapshots/source"
    provider = repo / "data/qlib/earnings"
    snapshot.mkdir(parents=True)
    calendar = pd.bdate_range("2024-01-01", "2024-01-10")
    build_qlib_store_from_frame(pd.DataFrame({"datetime": calendar, "instrument": "SOXX", "close": 10.0}),
                                provider_uri=reference, as_of="2024-01-10", field_map={"$close": "close"})
    sources = {}
    for s in SYMBOLS:
        path = snapshot / f"{s}.json"
        path.write_text(json.dumps(payload(s)))
        sources[s] = {"file": path.name, "sha256": sha256_file(path), "source_url": "https://example.test"}
    (snapshot / "snapshot_manifest.json").write_text(json.dumps({"as_of": "2024-01-10", "symbols": list(SYMBOLS), "sources": sources}))
    manifest = build_earnings_surprise_provider(snapshot, provider, reference, start="2024-01-01",
                                                 end="2024-01-10", as_of="2024-01-10")
    native, audit = load_earnings_surprise_features(provider, start="2024-01-01", end="2024-01-10")
    assert native.index.equals(calendar.rename("datetime"))
    assert native.loc["2024-01-04"].isna().all()
    assert np.allclose(native.loc["2024-01-05"].to_numpy(), [1 / 3] * 3)
    assert all(manifest[k] for k in ("diagnostic_only", "no_strict_pit", "no_promotion"))
    assert audit["readback"]["kernels"] == 1
    assert manifest["reference_provider"] == "data/qlib/reference"
    path = {"feature": provider / next(iter(manifest["feature_sha256"])),
            "calendar": provider / "calendars/day.txt",
            "raw_source": snapshot / "NVDA.json",
            "event_audit": snapshot / "events_availability.csv"}[tamper_target]
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_earnings_surprise_features(provider, start="2024-01-01", end="2024-01-10")
