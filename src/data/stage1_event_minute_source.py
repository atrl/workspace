"""Build a reproducible dual-price Qlib minute provider for event-time research."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.full_path_prices import _source_frames
from src.data.minute_price_basis import published_intraday_basis
from src.data.qlib_native import build_qlib_store_from_frame


FIELDS = ("open", "high", "low", "close", "raw_open", "raw_high",
          "raw_low", "raw_close", "volume", "raw_volume", "price_valid")


def build_event_minute_provider(*, source_manifest: Path, source_sha256: str,
                                provider_uri: Path, as_of: str, workspace_root: Path) -> dict:
    source_manifest, provider_uri = Path(source_manifest), Path(provider_uri)
    if hashlib.sha256(source_manifest.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("blocked: frozen NONE minute source manifest changed")
    if provider_uri.exists() and any(provider_uri.iterdir()):
        raise ValueError("blocked: event-minute provider already exists; verify it instead of rebuilding")
    receipt = json.loads(source_manifest.read_text())
    raw, events = _source_frames(source_manifest, receipt, as_of=as_of,
                                 workspace_root=workspace_root)
    source = pd.concat([raw["SOXL"], raw["SOXX"]], ignore_index=True)
    adjusted, basis = published_intraday_basis(source, events, as_of=as_of)
    adjusted["instrument"] = adjusted.symbol
    adjusted["datetime"] = adjusted.timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    if (adjusted.duplicated(["instrument", "datetime"]).any()
            or len(adjusted) != 1_560_120
            or adjusted[["open", "close", "raw_open", "raw_close"]].isna().any().any()):
        raise ValueError("blocked: adjusted event-minute source shape differs")
    manifest = build_qlib_store_from_frame(adjusted, provider_uri=provider_uri,
        freq="1min", region="us", as_of=as_of,
        field_map={field: field for field in FIELDS},
        source="verified NAS NONE full-minute plus published-affine event coefficients")
    manifest.update(source_manifest_sha256=source_sha256,
        source_schema=receipt["schema_version"], price_basis=basis,
        first_availability="right-edge final bar plus 60s research assumption; historical first receipt unqualified",
        fee_share_price="$raw_open", valuation_price="$open/$close",
        calendar_role="full observed RTH minute bars including one invalid source bar")
    (provider_uri / "qlib_store_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
