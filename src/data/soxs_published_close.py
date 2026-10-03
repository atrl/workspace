"""Qualified SOXS-only close features, on a separate published affine basis.

No model fitting, volume fabrication, legacy provider mutation, or price matching.
Futu raw closes have priority. Yahoo split-only closes fill whole missing dates.
A source-declared pre-Futu corporate-action supplement completes the event table.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.price_basis import PriceBasisError, _calendar_dates, _session_date, load_qlib_price_calendar
from src.data.published_price_basis import _events
from src.data.qlib_native import QlibProviderConfig, build_qlib_store_from_frame, init_qlib_provider

PROVIDER_RELATIVE = Path("data/qlib/soxs_published_close_20260925_v1")
SNAPSHOT_RELATIVE = Path("data/snapshots/soxs_published_close_20260925_v1")
BASIS = "soxs_published_affine_with_early_action_supplement_v1"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def compose_events(rehab: pd.DataFrame, supplemental: list[dict], yahoo_splits: dict, as_of: str) -> pd.DataFrame:
    """Validate event coverage and exact share counts, without estimating any value."""
    futu = _events(rehab)
    records = [{"date": row.event_date, "a": row.forward_adj_factorA,
                "b": row.forward_adj_factorB, "split": row.exact_split_price_factor,
                "source": "Futu.get_rehab"} for row in futu.itertuples()]
    first_futu = futu.event_date.min()
    for event in supplemental:
        date = _session_date(event["date"])
        if pd.isna(date) or date >= first_futu or not event.get("source_url") or not event.get("evidence_file"):
            raise PriceBasisError("supplement must be sourced, valid, and strictly before Futu events")
        records.append({"date": date, "a": float(event["a"]), "b": float(event["b"]),
                        "split": float(event["split"]), "source": event["source_url"]})
    events = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    if events.date.duplicated().any() or not np.isfinite(events[["a", "b", "split"]]).all().all():
        raise PriceBasisError("duplicate or nonfinite corporate actions")
    if (events.a <= 0).any() or (events.split <= 0).any() or (events.date > pd.Timestamp(as_of)).any():
        raise PriceBasisError("invalid or post-asof corporate actions")
    # Yahoo quotes reflect exact splits at retrieval vintage. Every observed split
    # must match the declared operator, including the missing 2011 reverse split.
    expected = {}
    for value in yahoo_splits.values():
        date = pd.Timestamp(value["date"], unit="s", tz="UTC").tz_convert("America/New_York").tz_localize(None).normalize()
        if date > pd.Timestamp(as_of):
            raise PriceBasisError("future Yahoo split makes quote vintage incompatible")
        ratio = float(value["denominator"]) / float(value["numerator"])
        if not np.isfinite(ratio) or ratio <= 0 or date in expected:
            raise PriceBasisError("invalid Yahoo split evidence")
        expected[date] = ratio
    actual = events.loc[events.split.ne(1)].set_index("date").split.to_dict()
    if actual != expected:
        raise PriceBasisError("published split table does not exactly match Yahoo quote split vintage")
    return events


def transform_soxs_close(futu_raw: pd.DataFrame, yahoo_quotes: pd.DataFrame,
                         events: pd.DataFrame, calendar: dict, as_of: str) -> tuple[pd.DataFrame, dict]:
    """Transform closes only; all source dates are checked before gap selection."""
    dates = _calendar_dates(calendar, as_of)
    for name, raw, date_column in (("Futu", futu_raw, "time_key"), ("Yahoo", yahoo_quotes, "date")):
        if date_column not in raw or "close" not in raw:
            raise PriceBasisError(f"{name}: missing date/close")
        observed = raw[date_column].map(_session_date)
        if observed.isna().any() or observed.duplicated().any() or not observed.isin(dates).all():
            raise PriceBasisError(f"{name}: invalid, duplicate, future, or off-calendar dates")
        values = pd.to_numeric(raw.close, errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise PriceBasisError(f"{name}: close must be observed, positive, and finite")
        if not {"open", "high", "low", "close"}.issubset(raw.columns):
            raise PriceBasisError(f"{name}: observed OHLC evidence is required")
        ohlc = raw[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        if (not np.isfinite(ohlc).all().all() or ohlc.le(0).any().any()
                or not (ohlc.low.le(ohlc.open) & ohlc.open.le(ohlc.high)
                        & ohlc.low.le(ohlc.close) & ohlc.close.le(ohlc.high)).all()):
            raise PriceBasisError(f"{name}: invalid observed OHLC ordering or values")
    if "code" not in futu_raw or not futu_raw.code.eq("US.SOXS").all():
        raise PriceBasisError("Futu raw source must contain only US.SOXS")
    primary = pd.DataFrame({"datetime": futu_raw.time_key.map(_session_date),
                            "observed_close": futu_raw.close.astype(float), "source": "Futu_NONE"})
    secondary = pd.DataFrame({"datetime": yahoo_quotes.date.map(_session_date),
                              "observed_close": yahoo_quotes.close.astype(float), "source": "Yahoo_SPLIT_ONLY"})
    gaps = secondary.loc[~secondary.datetime.isin(primary.datetime)]
    frame = pd.concat([primary, gaps], ignore_index=True).sort_values("datetime").reset_index(drop=True)
    if not pd.DatetimeIndex(frame.datetime).equals(dates):
        raise PriceBasisError("missing canonical sessions; no filling or shortened history allowed")
    aa, bb, ss = [], [], []
    for date in frame.datetime:
        a, b, split = 1.0, 0.0, 1.0
        for event in events.loc[events.date.gt(date)].itertuples():
            a *= event.a
            b = b * event.a + event.b
            split *= event.split
        aa.append(a); bb.append(b); ss.append(split)
    frame["affine_A"], frame["affine_B"], frame["split_price_factor"] = aa, bb, ss
    unadjusted = frame.observed_close.where(frame.source.eq("Futu_NONE"), frame.observed_close / frame.split_price_factor)
    frame["close"] = unadjusted * frame.affine_A + frame.affine_B
    if not np.isfinite(frame.close).all() or frame.close.le(0).any():
        raise PriceBasisError("affine output close must be positive finite")
    frame["instrument"] = "SOXS"
    audit = {"qualified": True, "basis": BASIS, "as_of": as_of, "rows": len(frame),
             "primary_rows": len(primary), "secondary_rows": len(gaps),
             "secondary_dates": gaps.datetime.dt.strftime("%Y-%m-%d").tolist(),
             "calendar_start": str(dates[0].date()), "calendar_end": str(dates[-1].date()),
             "calendar_sha256": calendar["calendar_sha256"], "market_metrics_used": False,
             "fields": ["$close"], "volume_used": False, "sdk_qfq_equivalence_claimed": False,
             "price_operator": "Chronological future events: A<-A*a; B<-B*a+b; close<-raw_close*A+B",
             "yahoo_undo": "quote close / product(exact future reverse-split price factors)",
             "source_priority": "Futu NONE, then Yahoo SPLIT_ONLY on missing whole dates only"}
    return frame, audit


def build_soxs_close_snapshot(root: Path, intake_dir: Path, *, supplemental_events: list[dict],
                              supplemental_evidence: list[str], as_of: str = "2026-09-25") -> dict:
    """Archive observed inputs, build the separate Qlib provider, and read it back."""
    root, intake_dir = Path(root), Path(intake_dir)
    snapshot, provider = root / SNAPSHOT_RELATIVE, root / PROVIDER_RELATIVE
    if snapshot.exists() or provider.exists():
        raise PriceBasisError("immutable supplemental snapshot already exists")
    intake = json.loads((intake_dir / "intake_manifest.json").read_text())
    if not intake["rehab_stable"] or intake["rehab_frame_sha256_before"] != intake["rehab_frame_sha256_after"]:
        raise PriceBasisError("rehabilitation changed during intake")
    for artifact in intake["raw_artifacts"]:
        if _sha_file(intake_dir / artifact["name"]) != artifact["sha256"]:
            raise PriceBasisError("intake archive hash mismatch")
    before = pd.read_csv(intake_dir / "rehab_before.csv")
    after = pd.read_csv(intake_dir / "rehab_after.csv")
    if not before.equals(after):
        raise PriceBasisError("rehabilitation source files differ")
    payload = json.loads((intake_dir / "yahoo_quote_and_splits_raw.json").read_text())
    result = payload["chart"]["result"][0]
    if "adjclose" in result["indicators"]:
        raise PriceBasisError("Yahoo adjusted close must not enter this source contract")
    quotes = pd.DataFrame(result["indicators"]["quote"][0])
    quotes["date"] = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert("America/New_York").strftime("%Y-%m-%d")
    events = compose_events(before, supplemental_events, result.get("events", {}).get("splits", {}), as_of)
    calendar = load_qlib_price_calendar(root / "data/qlib/published_futu_affine_20260925_v1", "2010-03-11", as_of)
    frame, audit = transform_soxs_close(pd.read_csv(intake_dir / "futu_none_all.csv"), quotes, events, calendar, as_of)
    snapshot.mkdir(parents=True)
    files = ["futu_none_all.csv", "rehab_before.csv", "rehab_after.csv", "yahoo_quote_and_splits_raw.json",
             "yahoo_early_event_check.json", "intake_manifest.json", *supplemental_evidence]
    for name in files:
        shutil.copy2(intake_dir / name, snapshot / name)
    _json_write(snapshot / "supplemental_events.json", supplemental_events)
    _json_write(snapshot / "canonical_calendar.json", calendar)
    events.to_csv(snapshot / "composed_events.csv", index=False)
    frame.to_csv(snapshot / "qualified_close_lineage.csv", index=False)
    source_manifest = {"source": "Futu NONE/RTH + Yahoo split-only gap quotes + published pre2013 actions",
                       "retrieved_at_utc": intake["ended_utc"], "as_of": as_of,
                       "source_intake": intake, "supplemental_events": supplemental_events,
                       "artifacts": [{"path": str(p.relative_to(root)), "sha256": _sha_file(p), "bytes": p.stat().st_size}
                                     for p in sorted(snapshot.iterdir()) if p.is_file()]}
    _json_write(snapshot / "source_manifest.json", source_manifest)
    qlib_manifest = build_qlib_store_from_frame(frame, provider_uri=provider, field_map={"$close": "close"},
                                               as_of=as_of, source=BASIS)
    from qlib.data import D
    init_qlib_provider(QlibProviderConfig(provider))
    returned = D.features(["SOXS"], ["$close"], start_time=audit["calendar_start"], end_time=as_of)
    series = returned.xs("SOXS", level="instrument")["$close"].sort_index()
    if not series.index.equals(pd.DatetimeIndex(frame.datetime, name="datetime")) or not np.array_equal(series.to_numpy(), frame.close.astype("float32").to_numpy()):
        raise PriceBasisError("native Qlib readback differs from qualified float32 closes")
    if not pd.DatetimeIndex(D.calendar()).equals(pd.DatetimeIndex(frame.datetime)):
        raise PriceBasisError("native Qlib calendar mismatch")
    audit.update(provider_uri=str(PROVIDER_RELATIVE), snapshot_uri=str(SNAPSHOT_RELATIVE),
                 raw_source_manifest_sha256=_sha_file(snapshot / "source_manifest.json"),
                 native_readback={"status": "passed", "rows": len(series), "storage_dtype": "float32",
                                  "max_relative_quantization": float(np.max(np.abs(series.to_numpy() / frame.close.to_numpy() - 1))),
                                  "source_api": ["qlib.init", "D.calendar", "D.instruments", "D.features"]},
                 source_limitations=["709 whole missing dates depend on Yahoo vendor quote quality, predominantly pre2013.",
                                     "Early distributions use Dividendhistory original cash amounts corroborated by Tiingo; Yahoo rounded implied cash amounts differ.",
                                     "Published affine is a research price convention, not a total-return index or SDK QFQ equivalence.",
                                     "Only close is qualified; Yahoo historical split-adjusted volumes often round to zero and are not used.",
                                     "Research-only qualification: all three series use a retrospective corporate-action vintage as of 2026-09-25, not archived point-in-time price vintages; additive future dividend affine B can change historical ratios. No strict PIT or production qualification is claimed."],
                 qlib_store_manifest=qlib_manifest,
                 provider_artifacts=[{"path": str(p.relative_to(provider)), "sha256": _sha_file(p)}
                                     for p in sorted(provider.rglob("*")) if p.is_file()])
    _json_write(provider / "soxs_close_manifest.json", audit)
    return audit


def load_qualified_soxs_close(root: Path) -> tuple[pd.Series, dict]:
    """Read qualified close through Qlib only, verifying all immutable artifacts."""
    root = Path(root)
    provider = root / PROVIDER_RELATIVE
    manifest = json.loads((provider / "soxs_close_manifest.json").read_text())
    if not manifest.get("qualified") or manifest.get("basis") != BASIS:
        raise PriceBasisError("unqualified SOXS close source")
    snapshot = root / manifest["snapshot_uri"]
    if _sha_file(snapshot / "source_manifest.json") != manifest["raw_source_manifest_sha256"]:
        raise PriceBasisError("source manifest hash mismatch")
    for artifact in json.loads((snapshot / "source_manifest.json").read_text())["artifacts"]:
        if _sha_file(root / artifact["path"]) != artifact["sha256"]:
            raise PriceBasisError("source artifact hash mismatch")
    for artifact in manifest["provider_artifacts"]:
        if _sha_file(provider / artifact["path"]) != artifact["sha256"]:
            raise PriceBasisError("provider artifact hash mismatch")
    init_qlib_provider(QlibProviderConfig(provider))
    from qlib.data import D
    dates = _calendar_dates(json.loads((snapshot / "canonical_calendar.json").read_text()), manifest["as_of"])
    if not pd.DatetimeIndex(D.calendar()).equals(dates):
        raise PriceBasisError("supplemental calendar mismatch")
    instruments = D.list_instruments(D.instruments("all"), as_list=True)
    if instruments != ["SOXS"]:
        raise PriceBasisError("supplemental instrument mismatch")
    frame = D.features(["SOXS"], ["$close"], start_time=manifest["calendar_start"], end_time=manifest["as_of"])
    series = frame.xs("SOXS", level="instrument")["$close"].sort_index().rename("SOXS")
    if not series.index.equals(dates) or not np.isfinite(series).all() or series.le(0).any():
        raise PriceBasisError("native SOXS close coverage or value qualification failed")
    return series, manifest
