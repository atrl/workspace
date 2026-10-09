"""Immutable full-download Cboe stores and explicitly lagged research inputs.

The source store preserves all supplied history through ``store_as_of``. A
separate, earlier ``market_as_of`` restricts feature consumption, not intake.
Historical revisions and exact publication vintages are not certified.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from src.data.qlib_native import build_qlib_store_from_frame


SYMBOLS = ("VIX9D", "VIX3M", "VVIX")
FEATURES = ("lag1_log_vix3m", "lag1_log_term", "lag1_log_vvix")
SCHEMA = "cboe_volatility_full_source_v1"
LIMITATIONS = [
    "Historical source vintages and original publication timestamps are not independently certified.",
    "Earlier backfilled rows and predecessor/index-launch identities are retained, not certified as contemporaneously published.",
    "Research excludes source observations before 2014-02-03; this conservative boundary does not resolve historical methodology or vintage uncertainty.",
    "Source-day 17:00 America/New_York availability is a conservative research assumption, not a verified historical release timestamp.",
]


def _require(value, message):
    if not value:
        raise ValueError("blocked: " + message)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _day(value, name):
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"blocked: invalid {name}") from error
    _require(not pd.isna(stamp) and stamp.tzinfo is None and stamp == stamp.normalize(),
             f"{name} must identify a finite naive calendar day")
    return stamp


def _read_source(symbol, raw, metadata, store_as_of):
    columns = ["DATE", "VVIX"] if symbol == "VVIX" else ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]
    try:
        source = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    except Exception as error:
        raise ValueError(f"blocked: unreadable {symbol} CSV") from error
    _require(list(source.columns) == columns and len(source) > 0, f"unexpected {symbol} source schema")
    _require(source.DATE.str.fullmatch(r"\d{2}/\d{2}/\d{4}").all(), f"invalid {symbol} DATE format")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(source.DATE, format="%m/%d/%Y", errors="raise"))
        values = source[columns[1:]].apply(pd.to_numeric, errors="raise").to_numpy(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"blocked: invalid {symbol} date or numeric values") from error
    _require(not dates.hasnans and not dates.has_duplicates and dates.is_monotonic_increasing,
             f"{symbol} dates must be unique and sorted; rows cannot be dropped or reordered")
    _require(np.isfinite(values).all() and (values > 0).all(), f"{symbol} values must be finite and positive")
    with np.errstate(over="ignore", under="ignore"):
        stored_close = values[:, -1].astype("float32")
    _require(np.isfinite(stored_close).all() and (stored_close > 0).all(),
             f"{symbol} close cannot be represented as positive finite Qlib float32")
    _require(dates[-1] == store_as_of, f"{symbol} full-download endpoint must equal store_as_of; no truncation")
    _require(isinstance(metadata, Mapping), f"{symbol} source metadata is required")
    urls = {}
    for key in ("requested_url", "final_url"):
        address = metadata.get(key, metadata.get("url"))
        parsed = urlparse(address) if isinstance(address, str) else None
        _require(parsed is not None and parsed.scheme == "https" and parsed.hostname is not None
                 and (parsed.hostname == "cboe.com" or parsed.hostname.endswith(".cboe.com")),
                 f"{symbol} requires official HTTPS Cboe source URLs")
        urls[key] = address
    if "sha256" in metadata:
        _require(metadata["sha256"] == _sha(raw), f"{symbol} download SHA-256 differs")
    if "bytes" in metadata:
        _require(metadata["bytes"] == len(raw), f"{symbol} download byte count differs")
    observed = metadata.get("retrieved_at_utc")
    try:
        retrieved = pd.Timestamp(observed)
    except (TypeError, ValueError) as error:
        raise ValueError(f"blocked: invalid {symbol} retrieval timestamp") from error
    _require(not pd.isna(retrieved) and retrieved.tzinfo is not None,
             f"{symbol} needs a timezone-aware retrieval timestamp")
    identity = {**urls, "retrieved_at_utc": retrieved.tz_convert("UTC").isoformat(),
        "sha256": _sha(raw), "bytes": len(raw), "rows": len(source),
        "date_range": [str(dates[0].date()), str(dates[-1].date())], "source_columns": columns,
        "snapshot_file": symbol + ".csv", "consumed_field": "VVIX" if symbol == "VVIX" else "CLOSE",
        "launch_limitations": metadata.get("launch_limitations", LIMITATIONS[1:3])}
    frame = pd.DataFrame({"datetime": dates, "instrument": symbol, "close": values[:, -1]})
    return frame, identity


def _native_readback(provider, expected=None, *, read_values=True):
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(freq="day", future=False))
    _require(len(calendar) > 0, "empty Cboe Qlib calendar")
    _require(sorted(D.list_instruments(D.instruments("all"), as_list=True)) == sorted(SYMBOLS),
             "Cboe native instrument inventory differs")
    if not read_values:
        return calendar, None
    native = D.features(list(SYMBOLS), ["$close"], start_time=calendar[0], end_time=calendar[-1], freq="day")
    if expected is not None:
        reference = expected.set_index(["instrument", "datetime"])["close"].astype("float32").sort_index()
        observed = native["$close"].dropna().sort_index()
        _require(observed.index.equals(reference.index)
                 and np.array_equal(observed.to_numpy(), reference.to_numpy()),
                 "full Cboe raw rows differ from native float32 readback")
    return calendar, native


def _verify_store(provider, manifest):
    _require(manifest.get("schema_version") == SCHEMA, "unsupported Cboe source manifest")
    _require(Path(manifest["provider_uri"]).resolve() == provider, "Cboe provider identity differs")
    for name, digest in manifest["provider_files_sha256"].items():
        path = provider / name
        _require(path.is_file() and _sha(path.read_bytes()) == digest, "Cboe provider bytes changed: " + name)
    snapshot = Path(manifest["snapshot_dir"])
    _require((snapshot / "manifest.json").is_file()
             and json.loads((snapshot / "manifest.json").read_text()) == manifest,
             "Cboe snapshot/provider manifests disagree")
    for item in manifest["sources"].values():
        path = snapshot / item["snapshot_file"]
        _require(path.is_file() and _sha(path.read_bytes()) == item["sha256"], "Cboe raw snapshot changed")
    return manifest


def intake_cboe_source(source_files, *, provider_uri, snapshot_dir, source_metadata, store_as_of):
    """Validate/copy every raw byte and build close-only native Qlib storage.

    Existing registered identical stores are read back; different or unknown
    destinations fail closed. No supplied row is removed at the market cutoff.
    """
    _require(isinstance(source_files, Mapping) and set(source_files) == set(SYMBOLS)
             and isinstance(source_metadata, Mapping) and set(source_metadata) == set(SYMBOLS),
             "exact VIX9D/VIX3M/VVIX source and metadata inventory required")
    endpoint = _day(store_as_of, "store_as_of")
    provider, snapshot = Path(provider_uri).resolve(), Path(snapshot_dir).resolve()
    _require(provider != snapshot and provider not in snapshot.parents and snapshot not in provider.parents,
             "provider and immutable raw snapshot must be separate directories")
    frames, sources, bytes_by_symbol = [], {}, {}
    for symbol in SYMBOLS:
        raw = Path(source_files[symbol]).read_bytes()
        frame, item = _read_source(symbol, raw, source_metadata[symbol], endpoint)
        frames.append(frame); sources[symbol] = item; bytes_by_symbol[symbol] = raw
    frame = pd.concat(frames, ignore_index=True)
    identity = {"schema_version": SCHEMA, "provider_uri": str(provider), "snapshot_dir": str(snapshot),
                "store_as_of": str(endpoint.date()), "sources": sources}
    manifest_path = provider / "cboe_source_manifest.json"
    if manifest_path.exists():
        retained = _verify_store(provider, json.loads(manifest_path.read_text()))
        _require(all(retained[key] == value for key, value in identity.items()), "immutable Cboe source identity differs")
        _native_readback(provider, frame)
        return retained
    _require(not provider.exists() or not any(provider.iterdir()), "refusing to overwrite an unregistered Cboe provider")
    _require(not (snapshot / "manifest.json").exists(), "orphaned snapshot manifest requires explicit recovery")
    for symbol, raw in bytes_by_symbol.items():
        destination = snapshot / (symbol + ".csv")
        _require(not destination.exists() or destination.read_bytes() == raw, "refusing to overwrite different raw snapshot bytes")
    snapshot.mkdir(parents=True, exist_ok=True)
    for symbol, raw in bytes_by_symbol.items():
        destination = snapshot / (symbol + ".csv")
        if not destination.exists():
            with destination.open("xb") as stream:
                stream.write(raw)
    store = build_qlib_store_from_frame(frame, provider_uri=provider, as_of=str(endpoint.date()),
        field_map={"close": "close"}, market="all", region="us", freq="day",
        source="immutable complete official Cboe downloads; no market-cutoff truncation")
    calendar, _ = _native_readback(provider, frame)
    files = ["calendars/day.txt", "instruments/all.txt", "qlib.yaml", "qlib_store_manifest.json",
             *[f"features/{symbol.lower()}/close.day.bin" for symbol in SYMBOLS]]
    manifest = {**identity, "field_map": {"close": "close"}, "raw_rows": len(frame),
        "rows_dropped": 0, "calendar_range": [str(calendar[0].date()), str(calendar[-1].date())],
        "native_fields": ["$close"], "synthetic_ohlc": False,
        "provider_files_sha256": {name: _sha((provider / name).read_bytes()) for name in files},
        "qlib_store_hash": store["hash"], "historical_vintage_certified": False,
        "limitations": LIMITATIONS, "source_api": ["qlib.init", "qlib.data.D.calendar", "qlib.data.D.features"]}
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (snapshot / "manifest.json").write_text(serialized)
    manifest_path.write_text(serialized)
    return manifest


def load_cboe_features(provider_uri, *, market_calendar, source_start="2014-02-03", market_as_of="2026-09-25"):
    """Lag on the complete benchmark calendar before selecting eligible rows.

    The caller supplies the audited benchmark Qlib calendar. Missing required
    source observations fail; neither forward filling nor row deletion occurs.
    This function initializes Qlib and must run outside an active Recorder.
    """
    provider = Path(provider_uri).resolve()
    manifest = _verify_store(provider, json.loads((provider / "cboe_source_manifest.json").read_text()))
    start, end = _day(source_start, "source_start"), _day(market_as_of, "market_as_of")
    _require(start >= pd.Timestamp("2014-02-03") and start < end, "source_start precedes conservative launch boundary or market endpoint")
    _require(end <= pd.Timestamp(manifest["store_as_of"]), "market_as_of exceeds immutable source store")
    calendar = pd.DatetimeIndex(market_calendar)
    _require(len(calendar) > 1 and not calendar.hasnans and calendar.tz is None
             and not calendar.has_duplicates and calendar.is_monotonic_increasing
             and calendar.equals(calendar.normalize()), "benchmark calendar must be complete ordered unique daily sessions")
    calendar = calendar[calendar <= end]
    _require(len(calendar) > 1 and calendar[-1] == end and calendar[0] <= start and start in calendar,
             "audited benchmark calendar must cover source_start and market_as_of exactly")
    from qlib.data import D

    native_calendar, _ = _native_readback(provider, read_values=False)
    _require(native_calendar[-1] == pd.Timestamp(manifest["store_as_of"]), "native source endpoint differs")
    # Explicit market boundary; later source rows remain stored and unconsumed.
    native = D.features(list(SYMBOLS), ["$close"], start_time=calendar[0], end_time=end, freq="day")
    close = native["$close"].unstack("instrument").reindex(index=calendar, columns=list(SYMBOLS)).astype(float)
    source_dates = pd.Series(calendar, index=calendar).shift(1)
    eligible = source_dates.ge(start)
    lagged = close.shift(1).loc[eligible]
    values = lagged.to_numpy(float)
    _require(len(lagged) > 0 and np.isfinite(values).all() and (values > 0).all(),
             "missing or invalid required lagged Cboe session; no fill or dropped rows allowed")
    result = pd.DataFrame(index=lagged.index)
    result.index.name = "datetime"
    result[FEATURES[0]] = np.log(lagged.VIX3M / 100.)
    result[FEATURES[1]] = np.log(lagged.VIX9D / lagged.VIX3M)
    result[FEATURES[2]] = np.log(lagged.VVIX / 100.)
    result["source_date"] = source_dates.loc[eligible].to_numpy()
    result["available_at"] = pd.DatetimeIndex(result.source_date).tz_localize("America/New_York") + pd.Timedelta(hours=17)
    result["decision_at"] = result.index.tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15)
    _require((result.available_at < result.decision_at).all(), "source availability must precede decision")
    audit = {"schema_version": SCHEMA, "provider_uri": str(provider), "store_as_of": manifest["store_as_of"],
        "market_as_of": str(end.date()), "source_start": str(start.date()), "lag_sessions": 1,
        "feature_columns": list(FEATURES), "rows": len(result), "rows_dropped_for_missingness": 0,
        "signal_range": [str(result.index[0].date()), str(result.index[-1].date())],
        "source_range": [str(result.source_date.iloc[0].date()), str(result.source_date.iloc[-1].date())],
        "market_calendar_sha256": _sha(calendar.asi8.tobytes()),
        "source_manifest_sha256": _sha((provider / "cboe_source_manifest.json").read_bytes()),
        "historical_vintage_certified": False, "availability_timestamp_verified": False,
        "limitations": LIMITATIONS, "source_api": ["qlib.init", "qlib.data.D.calendar", "qlib.data.D.features"]}
    result.attrs.update(historical_vintage_certified=False, availability_timestamp_verified=False)
    return result, audit
