"""Explicit raw-share and published-affine price planes for intraday research."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.minute_intake import atomic_parquet
from src.data.published_price_basis import _events


def normalize_none_history(raw: pd.DataFrame, *, symbol: str, interval: str,
                           fetched_at: str) -> pd.DataFrame:
    required = {"code", "time_key", "open", "high", "low", "close", "volume", "turnover"}
    if not required.issubset(raw) or not raw.code.eq("US." + symbol).all():
        raise ValueError("blocked: unexpected unadjusted source schema/instrument")
    if raw.duplicated(["code", "time_key"]).any():
        raise ValueError("blocked: duplicate unadjusted source observations")
    frame = raw[["open", "high", "low", "close", "volume", "turnover"]].copy()
    for field in frame:
        frame[field] = pd.to_numeric(frame[field], errors="coerce").astype(float)
    frame["timestamp"] = pd.to_datetime(raw.time_key).dt.tz_localize("America/New_York")
    frame["date"] = frame.timestamp.dt.strftime("%Y-%m-%d")
    frame["symbol"], frame["provider"], frame["interval"] = symbol, "futu", interval
    frame["session"], frame["adjusted"], frame["adjustment_type"] = "regular", False, "NONE"
    frame["source_tz"], frame["fetched_at"] = "America/New_York", fetched_at
    frame["price_valid"] = (
        np.isfinite(frame[["open", "high", "low", "close", "volume"]]).all(axis=1)
        & frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & frame.volume.ge(0)
        & frame.high.ge(frame[["open", "close", "low"]].max(axis=1) - 1e-8)
        & frame.low.le(frame[["open", "close", "high"]].min(axis=1) + 1e-8)
    )
    frame["bar_start_at"] = frame.timestamp - pd.to_timedelta(int(interval.removesuffix("m")), unit="min")
    frame["bar_end_at"] = frame.timestamp
    frame["received_at"] = pd.Timestamp(fetched_at)
    frame["is_final"] = frame.timestamp.dt.tz_convert("UTC") + pd.Timedelta(minutes=1) <= pd.Timestamp(fetched_at)
    # Historical retrieval time is recorded honestly; it is not first-public time.
    frame["availability_contract"] = "historical_right_edge_plus_60s_assumption_not_live_receipt_certification"
    return frame.sort_values("timestamp").reset_index(drop=True)


def published_intraday_basis(frame: pd.DataFrame, rehab_by_symbol: dict[str, pd.DataFrame],
                             *, as_of: str) -> tuple[pd.DataFrame, dict]:
    """Apply published event coefficients; never estimate them from prices/PnL.

    Raw observations, including malformed OHLC, remain unchanged in the source
    plane and retain a validity flag in the research plane. Dividends are price
    adjustments here; actual dividend cash-flow/PIT certification is separate.
    """
    if not frame.adjustment_type.eq("NONE").all() or frame.date.gt(as_of).any():
        raise ValueError("blocked: explicit NONE source through as_of required")
    result = frame.copy()
    receipts = {}
    for symbol, indices in result.groupby("symbol").groups.items():
        if symbol not in rehab_by_symbol:
            raise ValueError("blocked: published events unavailable for " + symbol)
        source = rehab_by_symbol[symbol]
        events = _events(source)
        events = events.loc[events.event_date.le(pd.Timestamp(as_of))]
        days = pd.DatetimeIndex(pd.to_datetime(result.loc[indices, "date"]))
        factor_a, factor_b, split = np.ones(len(indices)), np.zeros(len(indices)), np.ones(len(indices))
        for event in events.itertuples(index=False):
            before = days < event.event_date
            factor_a[before] *= event.forward_adj_factorA
            factor_b[before] = factor_b[before] * event.forward_adj_factorA + event.forward_adj_factorB
            split[before] *= event.exact_split_price_factor
        if not np.isfinite([factor_a, factor_b, split]).all() or (factor_a <= 0).any() or (split <= 0).any():
            raise ValueError("blocked: invalid published intraday operator")
        for field in ["open", "high", "low", "close"]:
            result.loc[indices, "raw_" + field] = result.loc[indices, field]
            result.loc[indices, field] = result.loc[indices, field].to_numpy() * factor_a + factor_b
        raw_volume = result.loc[indices, "volume"].to_numpy()
        raw_turnover = result.loc[indices, "turnover"].to_numpy()
        result.loc[indices, "raw_volume"] = raw_volume
        result.loc[indices, "raw_turnover"] = raw_turnover
        result.loc[indices, "volume"] = raw_volume / split
        result.loc[indices, "turnover"] = (factor_a * raw_turnover + factor_b * raw_volume) / split
        result.loc[indices, "affine_A"], result.loc[indices, "affine_B"] = factor_a, factor_b
        result.loc[indices, "split_price_factor"] = split
        receipts[symbol] = {"event_count": len(events), "nonzero_B_events": int(events.forward_adj_factorB.ne(0).sum()),
            "event_payload_sha256": hashlib.sha256(pd.util.hash_pandas_object(source, index=True).values.tobytes()).hexdigest()}
    result["adjusted"], result["adjustment_type"] = True, "PUBLISHED_FUTU_AFFINE"
    result["vwap"] = result.turnover / result.volume.replace(0, np.nan)
    vwap_valid = result.volume.gt(0) & np.isfinite(result.vwap)
    outside = vwap_valid & (result.vwap.lt(result.low * .999) | result.vwap.gt(result.high * 1.001))
    audit = {"basis": "published_futu_affine_v1", "as_of": as_of, "rows": len(result),
        "raw_ohlcv_retained": True, "invalid_source_rows": int((~result.price_valid).sum()),
        "positive_volume_rows": int(vwap_valid.sum()), "vwap_outside_rows": int(outside.sum()),
        "event_receipts": receipts, "price_operator": "NONE OHLC * chronological published A + B",
        "volume_operator": "NONE shares / exact published split factor",
        "turnover_operator": "(A * NONE turnover + B * NONE shares) / split factor",
        "SDK_QFQ_equivalence_claimed": False, "PIT_or_total_return_certified": False}
    return result, audit


def load_source_bundle(source_manifest: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    source_manifest = Path(source_manifest)
    manifest = json.loads(source_manifest.read_text())
    if not manifest.get("complete") or manifest.get("adjustment") != "NONE":
        raise ValueError("blocked: incomplete/unqualified source acquisition")
    parts, events = [], {}
    for entry in manifest["entries"]:
        path = source_manifest.parent / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("blocked: history source bytes changed")
        interval = {"K_1M": "1m", "K_5M": "5m"}[entry["ktype"]]
        parts.append(normalize_none_history(pd.read_parquet(path), symbol=entry["symbol"],
            interval=interval, fetched_at=entry["fetched_at"]))
    for entry in manifest["rehab"]:
        path = source_manifest.parent / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("blocked: published event source bytes changed")
        events[entry["symbol"]] = pd.read_parquet(path)
    return pd.concat(parts, ignore_index=True), events
