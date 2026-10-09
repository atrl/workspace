"""Qlib execution-price views of observed, right-edge intraday bars.

OHLC prices are copied unchanged. The additional execution-price field selects
the observed opening price on the first regular-session bar and the observed
close otherwise. It is a historical mark convention, not a live fill model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.qlib_native import build_qlib_store_from_frame


def build_repaired_intraday_view(*, cache_path: Path, repair_manifest: Path,
                                provider_uri: Path, start: str, as_of: str) -> dict:
    """Build a new provider from a frozen cache slice and observed repair bars.

    The original cache/provider is never edited. Official trading-day receipts
    distinguish complete regular and shortened sessions from partial captures.
    Repaired candles are new source vintages, not interpolated missing prices.
    """
    from src.data.intraday import normalize_futu_intraday_frame, resample_intraday_bars

    cache_path, repair_manifest = Path(cache_path).resolve(), Path(repair_manifest).resolve()
    receipt = json.loads(repair_manifest.read_text())
    calendar_meta = receipt["trading_calendar"]
    calendar_path = repair_manifest.parent / calendar_meta["path"]
    if _file_sha(calendar_path) != calendar_meta["sha256"]:
        raise ValueError("blocked: trading calendar receipt changed")
    calendar = json.loads(calendar_path.read_text())["days"]
    expected = {row["time"]: row["trade_date_type"] for row in calendar
                if start <= row["time"] <= as_of}
    before = _file_sha(cache_path)
    minute = pd.read_parquet(cache_path)
    if _file_sha(cache_path) != before:
        raise ValueError("blocked: source cache changed while snapshot was read")
    minute = minute[minute.date.ge(start) & minute.date.le(as_of)].copy()
    minute["timestamp"] = pd.to_datetime(minute["timestamp"], utc=True).dt.tz_convert("America/New_York")
    observed_revisions = []
    for entry in receipt["entries"]:
        path = repair_manifest.parent / entry["path"]
        if _file_sha(path) != entry["sha256"]:
            raise ValueError("blocked: repair receipt/object mismatch")
        raw = pd.read_parquet(path)
        repaired = normalize_futu_intraday_frame(raw, symbol=entry["symbol"],
                                                fetched_at=entry["fetched_at_utc"])
        repaired["timestamp"] = pd.to_datetime(repaired["timestamp"], utc=True).dt.tz_convert("America/New_York")
        if not repaired.date.eq(entry["date"]).all():
            raise ValueError("blocked: repaired bars have an unexpected date")
        mask = minute.symbol.eq(entry["symbol"]) & minute.date.eq(entry["date"])
        old = minute[mask]
        paired = old.merge(repaired, on=["symbol", "timestamp"], suffixes=("_old", "_new"), validate="one_to_one")
        observed_revisions.append({"symbol": entry["symbol"], "date": entry["date"],
            "previous_minute_rows": len(old), "observed_replacement_rows": len(repaired),
            "maximum_overlap_open_revision": float((paired.open_old - paired.open_new).abs().max()) if len(paired) else None,
            "scope": "observed historical source revision; not an interpolation or PIT guarantee"})
        minute = pd.concat([minute[~mask], repaired], ignore_index=True)
        minute["timestamp"] = pd.to_datetime(minute["timestamp"], utc=True).dt.tz_convert("America/New_York")
    validate_complete_minute_sessions(minute, expected)
    bars = resample_intraday_bars(minute, freq="15min", symbols=["SOXL", "SOXX"])
    for probe in receipt.get("direct_15m_probes", []):
        path = repair_manifest.parent / probe["path"]
        if _file_sha(path) != probe["sha256"]:
            raise ValueError("blocked: direct aggregate receipt changed")
        raw = pd.read_parquet(path)
        symbol = str(raw.code.iloc[0]).removeprefix("US.")
        direct = raw.assign(datetime=pd.to_datetime(raw.time_key).dt.tz_localize("America/New_York"))
        reconstructed = bars[bars.instrument.eq(symbol)].merge(direct, on="datetime", suffixes=("_rebuilt", "_direct"))
        if len(reconstructed) != len(raw):
            raise ValueError("blocked: direct aggregate timestamps differ")
        for key in ["open", "high", "low", "close", "volume"]:
            if not np.allclose(reconstructed[f"{key}_rebuilt"], reconstructed[f"{key}_direct"], rtol=0, atol=1e-7):
                raise ValueError(f"blocked: direct Futu 15min {key} disagrees with minute aggregation")
    manifest = build_qlib_store_from_frame(
        bars, provider_uri=provider_uri, freq="15min", as_of=as_of,
        field_map={key: key for key in ["open", "high", "low", "close", "volume", "bar_count"]},
        source=f"source cache snapshot {cache_path}; observed repair receipt {repair_manifest}",
    )
    manifest.update({"input_cache_sha256_at_read": before,
                     "repair_manifest_sha256": _file_sha(repair_manifest),
                     "trading_calendar_sha256": _file_sha(calendar_path),
                     "complete_session_validation": True, "expected_sessions": len(expected),
                     "observed_revisions": observed_revisions,
                     "timestamp_convention": "Futu minute right edges 09:31 onward; verified against direct K_15M aggregate receipts",
                     "purpose": "isolated research price repair; source cache and live providers unchanged"})
    (Path(provider_uri) / "qlib_store_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_uniform_intraday_view(*, source_manifest: Path, trading_calendar: Path,
                               provider_uri: Path, start: str, as_of: str) -> dict:
    """Build one fully observed, common-vintage 15-minute source snapshot."""
    source_manifest, trading_calendar = Path(source_manifest), Path(trading_calendar)
    receipt = json.loads(source_manifest.read_text())
    expected = {row["time"]: row["trade_date_type"] for row in json.loads(trading_calendar.read_text())["days"]
                if start <= row["time"] <= as_of}
    frames = []
    for entry in receipt["entries"]:
        if entry.get("ktype") != "K_15M" or entry.get("adjustment") != "QFQ":
            raise ValueError("blocked: unexpected uniform-vintage source protocol")
        path = source_manifest.parent / entry["path"]
        if _file_sha(path) != entry["sha256"]:
            raise ValueError("blocked: uniform-vintage source bytes changed")
        raw = pd.read_parquet(path)
        if not raw.code.eq(f"US.{entry['symbol']}").all():
            raise ValueError("blocked: source instrument identity mismatch")
        block = raw[["open", "high", "low", "close", "volume"]].copy()
        block["instrument"] = entry["symbol"]
        block["datetime"] = pd.to_datetime(raw.time_key)
        block["date"] = block.datetime.dt.strftime("%Y-%m-%d")
        for day, group in block.groupby("date"):
            if day not in expected or expected[day] not in {"WHOLE", "MORNING"}:
                raise ValueError("blocked: unexpected uniform-vintage session")
            close = "16:00" if expected[day] == "WHOLE" else "13:00"
            stamps = pd.date_range(f"{day} 09:45", f"{day} {close}", freq="15min")
            if not pd.DatetimeIndex(group.datetime).sort_values().equals(stamps):
                raise ValueError(f"blocked: incomplete uniform 15min session {entry['symbol']} {day}")
        if set(block.date) != set(expected):
            raise ValueError("blocked: uniform vintage omits required sessions")
        block["bar_count"] = block.groupby("date")["datetime"].transform("size")
        frames.append(block)
    if {str(frame.instrument.iloc[0]) for frame in frames} != {"SOXL", "SOXX"} or len(frames) != 2:
        raise ValueError("blocked: uniform vintage requires exactly both assets")
    panel = pd.concat(frames, ignore_index=True)
    values = panel[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("blocked: invalid uniform-vintage quote values")
    manifest = build_qlib_store_from_frame(panel, provider_uri=provider_uri, freq="15min", as_of=as_of,
        field_map={key: key for key in ["open", "high", "low", "close", "volume", "bar_count"]},
        source=str(source_manifest))
    manifest.update({"source_manifest_sha256": _file_sha(source_manifest),
                     "trading_calendar_sha256": _file_sha(trading_calendar),
                     "complete_session_validation": True, "expected_sessions": len(expected),
                     "single_source_vintage": True, "adjustment": "QFQ",
                     "purpose": "common observed price vintage for source-stability diagnosis; not PIT or broker-share qualification"})
    (Path(provider_uri) / "qlib_store_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def validate_complete_minute_sessions(frame: pd.DataFrame, calendar: dict[str, str]) -> None:
    """Reject missing sessions, partial captures and unexpected minute clocks."""
    for symbol in ("SOXL", "SOXX"):
        selected = frame[frame.symbol.eq(symbol)]
        if set(selected.date) != set(calendar):
            missing = sorted(set(calendar) - set(selected.date))
            raise ValueError(f"blocked: missing or unexpected minute sessions for {symbol}: {missing}")
        for day, block in selected.groupby("date"):
            kind = calendar[day]
            if kind not in {"WHOLE", "MORNING"}:
                raise ValueError(f"blocked: unknown US trading session type {kind}")
            close = "16:00" if kind == "WHOLE" else "13:00"
            expected = pd.date_range(f"{day} 09:31", f"{day} {close}", freq="1min", tz="America/New_York")
            actual = pd.DatetimeIndex(pd.to_datetime(block.timestamp)).sort_values()
            if not actual.equals(expected):
                raise ValueError(f"blocked: partial or duplicate minute capture for {symbol} on {day}")
            prices = block[["open", "high", "low", "close", "volume"]].to_numpy(float)
            if not np.isfinite(prices).all() or (prices[:, :4] <= 0).any() or (prices[:, 4] < 0).any():
                raise ValueError(f"blocked: invalid observed minute values for {symbol} on {day}")


def build_execution_mark_view(*, source_provider: Path, provider_uri: Path,
                              start: str, end: str, as_of: str,
                              symbols: tuple[str, ...] = ("SOXL", "SOXX")) -> dict:
    import qlib
    from qlib.data import D

    source_provider, provider_uri = Path(source_provider).resolve(), Path(provider_uri).resolve()
    if source_provider == provider_uri or source_provider in provider_uri.parents:
        raise ValueError("blocked: execution view must not overwrite its source")
    manifest_path = source_provider / "qlib_store_manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest.get("freq") != "15min":
        raise ValueError("blocked: execution view requires a 15min source")
    qlib.init(provider_uri=str(source_provider), region="us", kernels=1)
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    raw = D.features(list(symbols), fields, start_time=start, end_time=end, freq="15min").reset_index()
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    if raw.empty or raw.duplicated(["instrument", "datetime"]).any():
        raise ValueError("blocked: empty or duplicate source bars")
    raw = raw.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    raw.columns = [column.removeprefix("$") for column in raw.columns]
    opening = raw["datetime"].dt.strftime("%H:%M").eq("09:45")
    raw["execution_price"] = raw["close"].where(~opening, raw["open"])
    if not np.isfinite(raw.loc[opening, "execution_price"]).all():
        raise ValueError("blocked: missing observed session opens")
    manifest = build_qlib_store_from_frame(
        raw, provider_uri=provider_uri, region="us", freq="15min", as_of=as_of,
        field_map={key: key for key in ["open", "high", "low", "close", "volume", "execution_price"]},
        source=str(source_provider),
    )
    readback = D.features(list(symbols), ["$open", "$high", "$low", "$close", "$volume", "$execution_price"],
                          start_time=start, end_time=end, freq="15min").reset_index()
    readback = readback.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    original = raw[["instrument", "datetime", "open", "high", "low", "close", "volume"]]
    for column in ["open", "high", "low", "close", "volume"]:
        if not np.array_equal(original[column].to_numpy(), readback[f"${column}"].to_numpy(), equal_nan=True):
            raise ValueError(f"blocked: source {column} changed in execution view")
    manifest.update({
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_ohlcv_unchanged": True,
        "execution_price_rule": "09:45 right-edge bar uses its 09:30 open; all other bars use observed close",
        "price_clock": "right-edge 15-minute source labels; prices are observed historical marks",
        "live_fill_validation": False,
        "purpose": "native accounting reconciliation of prescribed opening and first-hour-close target schedules",
    })
    (provider_uri / "qlib_store_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def build_coherent_daily_view(*, daily_provider: Path, intraday_provider: Path,
                             provider_uri: Path, start: str, end: str, as_of: str,
                             symbols: tuple[str, ...] = ("SOXL", "SOXX")) -> dict:
    """Daily marks use actual intraday OHLCV once that source exists.

    The earlier daily history is retained. This makes the transition day's
    opening mark identical for the native daily prefix and intraday suffix;
    it never scales prices from either source to hide a discrepancy.
    """
    import qlib
    from qlib.data import D

    daily_provider, intraday_provider, provider_uri = [Path(p).resolve() for p in (daily_provider, intraday_provider, provider_uri)]
    for source in [daily_provider, intraday_provider]:
        if source == provider_uri or source in provider_uri.parents:
            raise ValueError("blocked: coherent daily view must not overwrite a source")
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    qlib.init(provider_uri=str(daily_provider), region="us", kernels=1)
    daily = D.features(list(symbols), fields, start_time=start, end_time=pd.Timestamp(end).normalize(), freq="day").reset_index()
    daily.columns = [column.removeprefix("$") for column in daily.columns]
    daily["datetime"] = pd.to_datetime(daily.datetime).dt.normalize()
    qlib.init(provider_uri=str(intraday_provider), region="us", kernels=1)
    intraday_calendar = pd.DatetimeIndex(D.calendar(freq="15min"))
    first_intraday_day = intraday_calendar.min().normalize()
    first = max(pd.Timestamp(start).normalize(), first_intraday_day)
    bars = D.features(list(symbols), fields, start_time=first,
                      end_time=pd.Timestamp(end).normalize() + pd.Timedelta(hours=16), freq="15min").reset_index()
    bars.columns = [column.removeprefix("$") for column in bars.columns]
    bars["datetime"] = pd.to_datetime(bars.datetime)
    bars = bars.sort_values(["instrument", "datetime"])
    if bars.empty or not np.isfinite(bars[["open", "high", "low", "close", "volume"]].to_numpy()).all():
        raise ValueError("blocked: incomplete observed intraday OHLCV for daily aggregation")
    bars["day"] = bars.datetime.dt.normalize()
    starts = bars.groupby(["instrument", "day"]).datetime.first()
    if not starts.dt.strftime("%H:%M").eq("09:45").all():
        raise ValueError("blocked: missing first regular-session bar for daily aggregation")
    aggregated = bars.groupby(["instrument", "day"], sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    ).reset_index().rename(columns={"day": "datetime"})
    retained = daily[daily.datetime < first_intraday_day].copy()
    native_days = set(aggregated.datetime)
    required = set(daily.loc[daily.datetime >= first_intraday_day, "datetime"])
    if not required.issubset(native_days):
        raise ValueError("blocked: intraday source omits a required daily session")
    combined = pd.concat([retained, aggregated], ignore_index=True).sort_values(["instrument", "datetime"])
    manifest = build_qlib_store_from_frame(
        combined, provider_uri=provider_uri, region="us", freq="day", as_of=as_of,
        field_map={key: key for key in ["open", "high", "low", "close", "volume"]},
        source=f"daily prefix {daily_provider}; observed intraday aggregates {intraday_provider}",
    )
    verification = D.features(list(symbols), ["$open"], start_time=start, end_time=end, freq="day").reset_index()
    expected = combined[["instrument", "datetime", "open"]]
    checked = verification.merge(expected, on=["instrument", "datetime"], validate="one_to_one")
    if len(checked) != len(expected) or not np.array_equal(checked["$open"].to_numpy(), checked.open.to_numpy(), equal_nan=True):
        raise ValueError("blocked: coherent daily opening-mark readback differs")
    manifest.update({
        "daily_source_manifest_sha256": hashlib.sha256((daily_provider / "qlib_store_manifest.json").read_bytes()).hexdigest(),
        "intraday_source_manifest_sha256": hashlib.sha256((intraday_provider / "qlib_store_manifest.json").read_bytes()).hexdigest(),
        "intraday_price_source_from": str(first_intraday_day.date()),
        "daily_prefix_rows": len(retained), "intraday_aggregate_rows": len(aggregated),
        "aggregation": "observed first open, max high, min low, last close, summed volume; no price scaling/interpolation",
        "purpose": "coherent native daily-prefix/intraday-suffix mark and quantity continuation",
    })
    (provider_uri / "qlib_store_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def session_price_marks(*, provider_uri: Path, sessions: pd.DataFrame,
                        symbols: tuple[str, ...] = ("SOXL", "SOXX"), freq: str = "15min") -> pd.DataFrame:
    """Read opening, first-hour-close and next-opening prices through Qlib D."""
    import qlib
    from qlib.data import D

    dates = sessions[["trade_date", "exit_date"]].apply(pd.to_datetime)
    if freq not in {"day", "15min"}:
        raise ValueError("blocked: unsupported native mark frequency")
    if dates.empty or dates.trade_date.isna().any() or dates.exit_date.isna().any():
        raise ValueError("blocked: incomplete session dates")
    qlib.init(provider_uri=str(Path(provider_uri).resolve()), region="us", kernels=1)
    raw = D.features(list(symbols), ["$open", "$close"], start_time=str(dates.trade_date.min()),
                     end_time=str(dates.exit_date.max() + pd.Timedelta(hours=16)), freq=freq)
    opens, closes = raw["$open"].unstack("instrument"), raw["$close"].unstack("instrument")
    output = dates.copy()
    for symbol in symbols:
        offset = pd.Timedelta(hours=9, minutes=45) if freq == "15min" else pd.Timedelta(0)
        opening = opens[symbol].reindex(pd.DatetimeIndex(dates.trade_date + offset)).to_numpy(float)
        cut = closes[symbol].reindex(pd.DatetimeIndex(dates.trade_date + pd.Timedelta(hours=10, minutes=30))).to_numpy(float) if freq == "15min" else np.full(len(dates), np.nan)
        exiting = opens[symbol].reindex(pd.DatetimeIndex(dates.exit_date + offset)).to_numpy(float)
        required_marks = np.r_[opening, cut, exiting] if freq == "15min" else np.r_[opening, exiting]
        if not np.isfinite(required_marks).all() or np.min(required_marks) <= 0:
            raise ValueError(f"blocked: missing or invalid native marks for {symbol}")
        output[f"{symbol.lower()}_open"] = opening
        output[f"{symbol.lower()}_cut"] = cut
        output[f"{symbol.lower()}_exit_open"] = exiting
        output[f"{symbol.lower()}_return"] = exiting / opening - 1.0
        output[f"early_ret_{symbol.lower()}"] = cut / opening - 1.0
    return output
