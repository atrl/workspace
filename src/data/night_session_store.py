"""Build a qlib provider for the collected night-session dataset.

Raw night-session parquet (intraday futures and premarket bars) is transformed
into a daily per-instrument feature frame and written to a qlib binary store
through ``build_qlib_store_from_frame``. After the build, the store is read back
through ``qlib.data.D`` so downstream cards never touch the raw parquet.

The store carries only causal, past-available daily fields:

- ``$return``: the regular-session 09:30-16:00 return for the instrument.
- ``$premarket``: the instrument's 04:00-09:30 premarket return.
- ``$nq_overnight``: NQ futures change from prior 18:00 ET to current 09:30 ET.
- ``$es_overnight``: ES futures change over the same overnight window.

The overnight fields are identical across instruments because they describe the
shared macro state; they are stored per row so a single ``D.features`` call
returns a self-contained aligned panel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import build_qlib_store_from_frame
from src.analysis.night_session_signal import (
    build_session_panel,
    load_night_session_instruments,
)

PROVIDER_URI = "data/qlib/night_session_20261010_v1"
RAW_DIR = "data/raw/external/night_session_20261010_v1"
SOURCE = "night_session_20261010_v1"


def build_night_session_frame(*, root: Path) -> pd.DataFrame:
    """Transform raw night-session parquet into a qlib-ready long frame."""
    instruments = load_night_session_instruments(root / RAW_DIR)
    panel = build_session_panel(instruments)
    rows = []
    for symbol in ("SOXL", "SOXX"):
        option = "soxl" if symbol == "SOXL" else "soxx"
        block = pd.DataFrame({
            "instrument": symbol,
            "datetime": panel.index,
            "open": panel[f"{option}_open"].to_numpy(float),
            "close": panel[f"{option}_close"].to_numpy(float),
            "return": panel[f"{option}_return"].to_numpy(float),
            "premarket": panel.get(f"{option}_premarket", pd.Series(np.nan, index=panel.index)).to_numpy(float),
            "nq_overnight": panel["nq_overnight"].to_numpy(float),
            "es_overnight": panel["es_overnight"].to_numpy(float),
        })
        rows.append(block.dropna(subset=["return"]))
    frame = pd.concat(rows, ignore_index=True)
    _require_finite(frame)
    return frame


def _require_finite(frame: pd.DataFrame) -> None:
    if frame.empty or not np.isfinite(frame["return"].to_numpy(float)).all():
        raise ValueError("blocked: night-session frame is empty or nonfinite")


def build_night_session_store(*, root: Path, provider_uri: str | Path | None = None) -> dict:
    frame = build_night_session_frame(root=root)
    uri = Path(provider_uri) if provider_uri is not None else (root / PROVIDER_URI)
    as_of = str(pd.to_datetime(frame["datetime"]).max().date())
    audit = build_qlib_store_from_frame(
        frame,
        provider_uri=uri,
        as_of=as_of,
        source=SOURCE,
    )
    return {"frame": frame, "audit": audit, "as_of": as_of}
