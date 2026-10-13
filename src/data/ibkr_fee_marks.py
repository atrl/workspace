"""Attach observed NONE prices to a frozen Qlib accounting-price view."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.qlib_native import QlibProviderConfig, build_qlib_store_from_frame, load_qlib_feature_frame


def build_ibkr_fee_marks(*, adjusted_provider: Path, raw_provider: Path, provider_uri: Path,
                         expected_adjusted_manifest: str, expected_raw_manifest: str,
                         start: str, as_of: str) -> dict:
    for source, expected in [(adjusted_provider, expected_adjusted_manifest), (raw_provider, expected_raw_manifest)]:
        if Path(source).resolve() == Path(provider_uri).resolve():
            raise ValueError("blocked: fee-price view cannot overwrite a source provider")
        path = Path(source) / "qlib_store_manifest.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("blocked: fee-price source manifest changed")
    fields = ["$open", "$high", "$low", "$close", "$volume", "$execution_price"]
    adjusted = load_qlib_feature_frame(QlibProviderConfig(adjusted_provider, freq="15min"),
        instruments=["SOXL", "SOXX"], fields=fields, start=start, end=as_of + " 16:00:00")
    raw = load_qlib_feature_frame(QlibProviderConfig(raw_provider, freq="1min"),
        instruments=["SOXL", "SOXX"], fields=["$open", "$close"], start=start, end=as_of + " 16:00:00")
    indices = adjusted.index
    dates = pd.DatetimeIndex(indices.get_level_values("datetime"))
    first = dates.strftime("%H:%M") == "09:45"
    lookup = dates.where(~first, dates.normalize() + pd.Timedelta(hours=9, minutes=31))
    raw_index = pd.MultiIndex.from_arrays([indices.get_level_values("instrument"), lookup], names=indices.names)
    prices = raw.reindex(raw_index)
    observed = np.where(first, prices["$open"], prices["$close"])
    if not np.isfinite(observed).all() or np.min(observed) <= 0:
        raise ValueError("needs_data: actual unadjusted opening/cut prices missing")
    adjusted["$raw_execution_price"] = observed
    manifest = build_qlib_store_from_frame(adjusted.reset_index(), provider_uri=provider_uri,
        field_map={field: field for field in adjusted}, freq="15min", as_of=as_of,
        source="frozen native accounting OHLCV plus observed raw NONE execution prices for IBKR share-based fees")
    manifest.update(adjusted_source_manifest_sha256=expected_adjusted_manifest,
                    raw_source_manifest_sha256=expected_raw_manifest,
                    raw_execution_price_rule="09:45 uses actual NONE 09:31 bar open; later 15min labels use actual NONE close",
                    future_affine_coefficients_used_to_infer_raw_prices=False,
                    nav_share_units="inherited adjusted Qlib units",
                    commission_share_units="integer trade notional divided by actual NONE quote",
                    actual_broker_holding_dividend_or_fill_parity=False)
    (Path(provider_uri) / "qlib_store_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
