"""Offline initial-release yields and explicit information availability.

Yield values may be negative. Unknown observations and premature publication
dates remain unknown; they are never replaced by an older valid observation.
These are information fields, not executable bond or ETF prices.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def read_initial_yield_archive(directory, *, as_of, series_id):
    """Read a pinned official archive, allowing finite negative real yields."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    _require(manifest["series_id"] == series_id and manifest["as_of"] == as_of
             and manifest["output_format"] == "Observations, Initial Release Only", "source format or cutoff differs")
    frames = []
    for entry in manifest["entries"]:
        path = (directory.parent / entry["path"]).resolve()
        _require(path.is_relative_to(directory.resolve()), "archive path escapes source directory")
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == entry["sha256"], "initial-release bytes differ")
        archive = zipfile.ZipFile(io.BytesIO(raw))
        readme = archive.read("README.txt").decode()
        _require("Output Format: Observations, Initial Release Only" in readme
                 and "Series ID: " + series_id in readme, "official archive identity differs")
        frame = pd.read_csv(io.BytesIO(archive.read(entry["csv_name"])), na_values=["."])
        _require(list(frame.columns) == ["period_start_date", series_id, "realtime_start_date"], "initial-release schema differs")
        frame = frame.rename(columns={"period_start_date": "observation_date", series_id: "value",
                                      "realtime_start_date": "release_date"})
        frame["value"] = pd.to_numeric(frame.value, errors="raise")
        frame["batch_sha256"] = entry["sha256"]
        frames.append(frame)
    _require(bool(frames), "source archive inventory is empty")
    result = pd.concat(frames, ignore_index=True).sort_values("observation_date").reset_index(drop=True)
    for column in ("observation_date", "release_date"):
        result[column] = pd.to_datetime(result[column], errors="raise")
        _require(result[column].notna().all() and (result[column] <= pd.Timestamp(as_of)).all(),
                 "unknown or after-cutoff raw date")
    _require(not result.observation_date.duplicated().any(), "duplicate initial-release observation")
    _require(np.isfinite(result.loc[result.value.notna(), "value"]).all(), "nonfinite published yield")
    result["series_id"] = series_id
    return result


def qualify_yield_change(raw, calendar, *, change_sessions):
    """Use only released observations, then one fixed native-session difference.

Availability is next native 09:31 New York after both observation and release
dates. A last-published state carried forward is labeled as such; no synthetic
daily observations are generated. Unknown latest observations invalidate it.
"""
    _require(isinstance(change_sessions, int) and not isinstance(change_sessions, bool)
             and change_sessions > 0, "change_sessions requires a positive integer")
    calendar = pd.DatetimeIndex(calendar)
    _require(not calendar.empty and not calendar.hasnans and not calendar.has_duplicates
             and calendar.is_monotonic_increasing and calendar.tz is None, "complete native calendar required")
    events = raw.copy()
    _require({"observation_date", "release_date", "value"}.issubset(events.columns), "source event schema differs")
    for name in ("observation_date", "release_date"):
        dates = pd.DatetimeIndex(pd.to_datetime(events[name], errors="raise"))
        _require(not dates.hasnans and dates.tz is None and dates.equals(dates.normalize()),
                 "source dates must be known date-level publication observations")
        events[name] = dates
    _require(np.isfinite(events.loc[events.value.notna(), "value"]).all(), "nonfinite published yield")
    _require(not events.observation_date.duplicated().any(), "duplicate source observation")
    events["on_native_session"] = events.observation_date.isin(calendar)
    events["premature_release"] = events.release_date < events.observation_date
    events["qualified"] = events.on_native_session & events.value.notna() & ~events.premature_release
    anchor = pd.concat([events.observation_date, events.release_date], axis=1).max(axis=1)
    next_position = calendar.searchsorted(pd.DatetimeIndex(anchor), side="right")
    valid = next_position < len(calendar)
    available = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns, UTC]")
    available.loc[valid] = (calendar[next_position[valid]].tz_localize("America/New_York")
                            + pd.Timedelta(hours=9, minutes=31)).tz_convert("UTC")
    events["available_at"] = available
    usable_clock = events.loc[events.on_native_session & events.available_at.notna()]
    rows = []
    for day in calendar:
        decision = (day.tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15)).tz_convert("UTC")
        published = usable_clock.loc[usable_clock.available_at <= decision]
        row = dict(datetime=day, state_known=0., real_yield=np.nan, observation_date=pd.NaT,
                   available_at=pd.NaT, availability_calendar_index=np.nan,
                   reason="no_source_yet", decision_at=decision)
        if not published.empty:
            event = published.sort_values("observation_date").iloc[-1]
            availability_day = event.available_at.tz_convert("America/New_York").tz_localize(None).normalize()
            row.update(observation_date=event.observation_date, available_at=event.available_at,
                       availability_calendar_index=int(calendar.get_indexer([availability_day])[0]),
                       reason="last_published_state" if event.qualified else "latest_observation_unknown_or_quarantined")
            if event.qualified:
                row.update(state_known=1., real_yield=float(event.value))
        rows.append(row)
    state = pd.DataFrame(rows)
    endpoints_known = state.state_known.eq(1) & state.state_known.shift(change_sessions).eq(1)
    state["known"] = endpoints_known.astype(float)
    state["real_yield_change"] = (state.real_yield - state.real_yield.shift(change_sessions)).where(endpoints_known)
    state["availability_calendar_index"] = state.availability_calendar_index.where(endpoints_known)
    audit = dict(raw_rows=len(raw), calendar_rows=len(calendar), known_signal_rows=int(state.known.sum()),
                 unknown_signal_rows=int(state.known.eq(0).sum()),
                 premature_market_observations=int((events.premature_release & events.on_native_session).sum()),
                 non_session_raw_observations=int((~events.on_native_session).sum()),
                 negative_yield_observations=int((events.value < 0).sum()),
                 change_sessions=change_sessions, value_unit="percentage_points", missing_values_filled=False,
                 calendar_rows_dropped=0, date_level_initial_release_verified=True,
                 intraday_first_public_timestamp_verified=False, source_deployment_gate=False,
                 availability_contract="next native session09:31NY after max(observation,initial-release dates)",
                 state_carry="last publicly released observation only; unknown latest invalidates state")
    return state, events, audit


def build_yield_information_store(state, *, provider_uri, as_of, source, instrument):
    """Build and read back every calendar row through native Qlib APIs."""
    from src.data.qlib_native import build_qlib_store_from_frame
    from src.data.provider_content_hash import provider_content_hash
    import qlib
    from qlib.data import D

    fields = ["known", "real_yield", "real_yield_change", "availability_calendar_index"]
    frame = state[["datetime", *fields]].copy()
    frame["instrument"] = instrument
    provider_uri = Path(provider_uri).resolve()
    manifest = build_qlib_store_from_frame(frame, provider_uri=provider_uri, as_of=as_of,
        source=source, field_map={name: name for name in fields}, region="us")
    manifest.update(information_instrument=True, not_trading_prices=True, initial_release_values_only=True,
                    date_level_vintage_verified=True, intraday_publication_timestamp_verified=False,
                    source_deployment_gate=False, value_unit="percentage_points", negative_real_yields_valid=True)
    (provider_uri / "qlib_store_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    qlib.init(provider_uri=str(provider_uri), region="us", kernels=1)
    _require(pd.DatetimeIndex(D.calendar(freq="day")).equals(pd.DatetimeIndex(state.datetime)),
             "information store changed native calendar")
    readback = D.features([instrument], ["$" + name for name in fields], freq="day")
    for name in fields:
        _require(np.allclose(readback["$" + name], state[name], rtol=0, atol=1e-6, equal_nan=True),
                 "native information readback differs: " + name)
    return manifest, readback, provider_content_hash(provider_uri)[0]
