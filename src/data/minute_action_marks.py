"""Observed minute features and a sparse native execution-event calendar."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.minute_price_basis import normalize_none_history, published_intraday_basis
from src.data.qlib_native import QlibProviderConfig, build_qlib_store_from_frame, load_qlib_feature_frame


def first_hour_features(raw):
    """Use exactly the completed 09:31..10:30 NONE bars, never later prices."""
    raw = raw.copy()
    stamps = pd.DatetimeIndex(raw.index.get_level_values("datetime"))
    raw = raw.loc[(stamps.hour * 60 + stamps.minute >= 571) & (stamps.hour * 60 + stamps.minute <= 630)]
    rows = []
    for day, block in raw.groupby(pd.DatetimeIndex(raw.index.get_level_values("datetime")).normalize()):
        expected = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=60, freq="1min")
        assets = {symbol: block.xs(symbol, level="instrument").reindex(expected) for symbol in ("SOXL", "SOXX")}
        if any(not np.isfinite(b[["$open", "$close", "$volume"]]).all().all() or b.index.size != 60
               or ("$price_valid" in b and not b["$price_valid"].eq(1).all())
               or (b[["$open", "$close"]] <= 0).any().any() for b in assets.values()):
            continue
        returns = {s: np.diff(np.log(np.r_[b["$open"].iloc[0], b["$close"]])) for s, b in assets.items()}
        soxx, soxl = assets["SOXX"], assets["SOXL"]
        r = returns["SOXX"]
        path = np.r_[soxx["$open"].iloc[0], soxx["$close"].to_numpy()]
        variance = max(float(r @ r), 1e-20)
        volume = max(float(soxx["$volume"].sum()), 1e-20)
        rows.append({"trade_date": day, "minute_return60": float(r.sum()),
            "minute_return15": float(r[-15:].sum()), "minute_log_rv": float(np.log(variance)),
            "minute_downside_share": float(r[r < 0] @ r[r < 0] / variance),
            "minute_late_rv_share": float(r[-15:] @ r[-15:] / variance),
            "minute_efficiency": float(abs(r.sum()) / max(np.abs(r).sum(), 1e-20)),
            "minute_drawdown": float((path / np.maximum.accumulate(path) - 1).min()),
            "minute_rebound": float(path[-1] / path.min() - 1),
            "minute_late_volume_share": float(soxx["$volume"].iloc[-15:].sum() / volume),
            "minute_tracking_return": float((returns["SOXL"] - 3 * r).sum()),
            "minute_tracking_log_rv": float(np.log(max(float(np.sum((returns["SOXL"] - 3 * r) ** 2)), 1e-20))),
            "minute_sign_agreement": float(np.mean(np.sign(returns["SOXL"]) == np.sign(r))),
            "feature_available_at": day + pd.Timedelta(hours=10, minutes=31)})
    if not rows:
        return pd.DataFrame(columns=["feature_available_at"], index=pd.DatetimeIndex([], name="trade_date"))
    return pd.DataFrame(rows).set_index("trade_date")


def build_action_marks(*, source_manifest, source_sha256, raw_provider, raw_manifest_sha256,
                       provider_uri, start, as_of):
    source_manifest, raw_provider = Path(source_manifest), Path(raw_provider)
    for path, digest in [(source_manifest, source_sha256), (raw_provider / "qlib_store_manifest.json", raw_manifest_sha256)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("blocked: frozen minute source changed")
    receipt = json.loads(source_manifest.read_text())
    parts, events = [], {}
    for symbol in ("SOXL", "SOXX"):
        entry = next(e for e in receipt["entries"] if e["symbol"] == symbol)
        event = next(e for e in receipt["rehab"] if e["symbol"] == symbol)
        for item in (entry, event):
            path = source_manifest.parent / item["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("blocked: observed quote/event receipt changed")
        raw = normalize_none_history(pd.read_parquet(source_manifest.parent / entry["path"]),
                                    symbol=symbol, interval="1m", fetched_at=entry["fetched_at"])
        raw = raw.loc[raw.date.ge(start) & raw.date.le(as_of)].copy()
        clock = raw.timestamp.dt.strftime("%H:%M")
        # Actual 09:30 and 10:32 opens; actual normal/half-day session closes.
        day_last = raw.groupby("date").timestamp.transform("max")
        selected = raw.loc[clock.isin(["09:31", "10:33"]) | raw.timestamp.eq(day_last)].copy()
        selected["execution_at"] = selected.timestamp.where(selected.timestamp.eq(day_last), selected.bar_start_at)
        selected["is_close_mark"] = selected.timestamp.eq(day_last)
        if not selected.price_valid.all() or not selected.groupby("date").size().eq(3).all():
            raise ValueError("needs_data: execution event minute missing or invalid")
        parts.append(selected)
        events[symbol] = pd.read_parquet(source_manifest.parent / event["path"])
    published, basis = published_intraday_basis(pd.concat(parts, ignore_index=True), events, as_of=as_of)
    published["instrument"] = published.symbol
    published["datetime"] = published.execution_at.dt.tz_localize(None)
    published["$execution_price"] = np.where(published.is_close_mark, published.close, published.open)
    published["$raw_execution_price"] = np.where(published.is_close_mark, published.raw_close, published.raw_open)
    # This is a point-mark series, not fabricated intraday OHLC bars.
    for field in ("$open", "$high", "$low", "$close"):
        published[field] = published["$execution_price"]
    published["$volume"] = published.volume
    fields = ["$open", "$high", "$low", "$close", "$volume", "$execution_price", "$raw_execution_price"]
    manifest = build_qlib_store_from_frame(published, provider_uri=provider_uri, freq="1min", as_of=as_of,
        field_map={field: field for field in fields}, source="NAS observed minute opens/session-close execution point marks")
    manifest.update(source_manifest_sha256=source_sha256, basis=basis, calendar_role="sparse_execution_events_not_complete_1min_bar_history",
        feature_cut="10:30 ET close; historical plus60s availability; decision10:31; execution10:32 actual open",
        opening_execution="09:30 actual open, prior-session source known before09:29", synthetic_prices=False,
        actual_first_receipt_or_broker_fill_qualified=False)
    (Path(provider_uri) / "qlib_store_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    marks = load_qlib_feature_frame(QlibProviderConfig(Path(provider_uri), freq="1min"), instruments=["SOXL", "SOXX"],
        fields=fields, start=start, end=as_of + " 16:00:00")
    raw = load_qlib_feature_frame(QlibProviderConfig(raw_provider, freq="1min"), instruments=["SOXL", "SOXX"],
        fields=["$open", "$close", "$volume", "$price_valid"], start=start, end=as_of + " 16:00:00")
    return marks, first_hour_features(raw), manifest
