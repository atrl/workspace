"""Build a qlib 60-minute store for the NQ futures raw capture.

The store carries ONLY OHLCV. Bar clock metadata (``bar_start``, ``bar_end``,
``available_at``) is NOT stored; it is derived from the qlib 60-minute calendar
index at read time, so nothing assumes a bar is available before it closes.

Raw parquet is a source-audit artifact; research reads through ``qlib.data.D``.
A duplicate raw timestamp or a source without an explicitly-declared timezone is
blocked rather than quietly coerced.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.qlib_native import build_qlib_store_from_frame

PROVIDER_URI = "data/qlib/nq_futures_60min_20261010_v1"
RAW = "data/raw/external/night_session_20261010_v1/NQ_F.parquet"
INSTRUMENT = "NQ"
FREQ = "60min"
FIELDS = ("open", "high", "low", "close", "volume")


def build_nq_futures_frame(*, root: Path) -> pd.DataFrame:
    raw = pd.read_parquet(root / RAW)
    raw.index = pd.to_datetime(raw.index)
    if raw.index.has_duplicates:
        raise ValueError("blocked: duplicate raw NQ timestamps")
    if raw.index.tz is None:
        raise ValueError("blocked: raw NQ timezone must be explicitly declared, not assumed NY")
    # Explicit source timezone -> naive America/New_York for the qlib store.
    raw.index = raw.index.tz_convert("America/New_York").tz_localize(None)
    raw = raw.sort_index()
    frame = pd.DataFrame({
        "instrument": INSTRUMENT, "datetime": raw.index,
        "open": raw["open"].astype(float), "high": raw["high"].astype(float),
        "low": raw["low"].astype(float), "close": raw["close"].astype(float),
        "volume": raw["volume"].astype(float),
    })
    return frame


def build_nq_futures_store(*, root: Path, provider_uri: str | Path | None = None) -> dict:
    frame = build_nq_futures_frame(root=root)
    uri = Path(provider_uri) if provider_uri is not None else (root / PROVIDER_URI)
    as_of = str(pd.to_datetime(frame["datetime"]).max().date())
    manifest = build_qlib_store_from_frame(
        frame, provider_uri=uri, freq=FREQ, as_of=as_of, source="night_session_20261010_v1",
        field_map={name: name for name in FIELDS},
    )
    return {"frame": frame, "manifest": manifest, "as_of": as_of, "provider_uri": str(uri)}
