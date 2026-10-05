"""Frozen VIX closes for causal legacy-rule research, read through Qlib only.

Yahoo's historical chart is an as-retrieved history, not verified publication
vintages. The stated 17:00 New York availability is a research assumption.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.qlib_native import (
    QlibProviderConfig, build_qlib_store_from_frame, init_qlib_provider,
    load_qlib_feature_frame,
)

CANONICAL_PROVIDER = Path("data/qlib/published_futu_affine_20260925_v1")
VIX_PROVIDER = Path("data/qlib/legacy_vix_20260925_v1")
VIX_SNAPSHOT = Path("data/snapshots/legacy_vix_20260925_v1")
START = "2010-03-11"
AS_OF = "2026-09-25"
AVAILABILITY = {
    "assumed_available_local_time": "17:00",
    "timezone": "America/New_York",
    "execution_eligibility": "next canonical session open only",
    "historical_publication_vintage_verified": False,
    "limitation": "As-retrieved Yahoo historical closes; historical revisions/publication vintages not verified.",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _calendar(root: Path) -> tuple[pd.DatetimeIndex, dict]:
    from qlib.data import D

    provider = root / CANONICAL_PROVIDER
    raw = (provider / "qlib_store_manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest["calendar_start"] != START or manifest["calendar_end"] != AS_OF or manifest["as_of"] != AS_OF:
        raise ValueError("blocked: canonical price-provider range changed")
    init_qlib_provider(QlibProviderConfig(provider_uri=provider))
    calendar = pd.DatetimeIndex(D.calendar(freq="day"), name="datetime")
    if calendar.empty or calendar[0] != pd.Timestamp(START) or calendar[-1] != pd.Timestamp(AS_OF):
        raise ValueError("blocked: native canonical calendar does not match frozen bounds")
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("blocked: canonical calendar is not unique and ordered")
    dates = calendar.strftime("%Y-%m-%d").tolist()
    return calendar, {"provider_uri": CANONICAL_PROVIDER.as_posix(), "manifest_sha256": _sha(raw),
                      "calendar_sha256": _sha("\n".join(dates).encode()), "calendar_rows": len(calendar),
                      "calendar_start": START, "calendar_end": AS_OF, "source_api": "qlib.data.D.calendar"}


def parse_yahoo_vix_close(raw: bytes, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Validate bounded raw closes before projecting onto the Qlib calendar.

    Null Yahoo holiday placeholders outside the equity calendar are archived and
    explicitly excluded; a missing or invalid equity-session close fails closed.
    Current summary metadata is never used as historical observations.
    """
    value = json.loads(raw)
    chart = value.get("chart", {})
    if chart.get("error") or len(chart.get("result") or []) != 1:
        raise ValueError("blocked: Yahoo VIX response is absent or has an error")
    result = chart["result"][0]
    meta = result["meta"]
    if meta.get("symbol") != "^VIX" or meta.get("instrumentType") != "INDEX" or meta.get("dataGranularity") != "1d":
        raise ValueError("blocked: Yahoo VIX symbol, instrument type, or frequency mismatch")
    if "adjclose" in result["indicators"]:
        raise ValueError("blocked: adjusted-close response is not the requested close-only source")
    stamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    if len(stamps) != len(closes) or not stamps:
        raise ValueError("blocked: VIX timestamps and closes do not align")
    dates = pd.to_datetime(stamps, unit="s", utc=True).tz_convert(meta["exchangeTimezoneName"]).tz_localize(None).normalize()
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("blocked: duplicate or unordered VIX dates")
    if dates.min() < calendar.min() or dates.max() > calendar.max():
        raise ValueError("blocked: VIX raw observations exceed frozen calendar bounds")
    series = pd.Series(closes, index=dates, dtype=float)
    missing = calendar.difference(dates)
    aligned = series.reindex(calendar)
    if len(missing) or not np.isfinite(aligned.to_numpy()).all() or (aligned <= 0).any():
        raise ValueError("blocked: missing, nonfinite, or nonpositive canonical-session VIX close; no filling permitted")
    extras = dates.difference(calendar)
    frame = pd.DataFrame({"datetime": calendar, "instrument": "VIX", "close": aligned.to_numpy()})
    return frame, {
        "source_rows": len(series), "canonical_rows": len(frame), "missing_canonical_sessions": 0,
        "excluded_noncanonical_dates": extras.strftime("%Y-%m-%d").tolist(),
        "excluded_noncanonical_null_closes": int(series.reindex(extras).isna().sum()),
        "source_timezone": meta["exchangeTimezoneName"], "used_fields": ["timestamp", "indicators.quote[0].close"],
        "ignored_fields": "All summary metadata and OHLCV fields other than historical close; metadata may postdate as_of.",
        "adjustment_policy": "raw quote.close; no adjusted-close input, split adjustment, normalization, or missing-value filling",
    }


def freeze_legacy_vix_snapshot(root: str | Path, raw_path: str | Path, *, source_url: str, retrieved_at: str) -> dict:
    """Freeze a new source archive/provider, refusing to overwrite either one."""
    root = Path(root).resolve()
    provider, snapshot = root / VIX_PROVIDER, root / VIX_SNAPSHOT
    if provider.exists() or snapshot.exists():
        raise ValueError("blocked: VIX snapshot/provider already exists and is immutable")
    calendar, reference = _calendar(root)
    raw = Path(raw_path).read_bytes()
    frame, qualification = parse_yahoo_vix_close(raw, calendar)
    snapshot.mkdir(parents=True, exist_ok=False)
    provider.mkdir(parents=True, exist_ok=False)
    (snapshot / "yahoo_vix_chart.json").write_bytes(raw)
    _write_json(snapshot / "status.json", {"status": "building"})
    try:
        built = build_qlib_store_from_frame(frame, provider_uri=provider, as_of=AS_OF,
            field_map={"close": "close"}, source="Yahoo chart ^VIX raw quote.close; immutable source archive")
        values, native = _read_native(provider, calendar)
        expected = frame["close"].to_numpy(dtype=np.float32)
        if not np.array_equal(values.to_numpy(), expected):
            raise ValueError("blocked: native VIX readback differs from source float32 values")
        files = {str(p.relative_to(root)): _sha(p.read_bytes()) for p in sorted(provider.rglob("*")) if p.is_file()}
        files[str(VIX_SNAPSHOT / "yahoo_vix_chart.json")] = _sha(raw)
        manifest = {
            "schema_version": "legacy_vix_snapshot_v1", "status": "qualified", "as_of": AS_OF,
            "provider_uri": VIX_PROVIDER.as_posix(), "source_archive": str(VIX_SNAPSHOT / "yahoo_vix_chart.json"),
            "source_url": source_url, "retrieved_at": retrieved_at, "raw_sha256": _sha(raw),
            "source_request": {"symbol": "^VIX", "interval": "1d", "includeAdjustedClose": False,
                               "start_inclusive": START, "end_exclusive": "2026-09-26"},
            "reference_calendar": reference, "qualification": qualification,
            "availability": AVAILABILITY, "native_readback": native,
            "max_absolute_float32_quantization": float(np.max(np.abs(expected.astype(float) - frame["close"].to_numpy()))),
            "qlib_manifest_hash": built["hash"], "file_sha256": files,
            "qlib_docs": "https://qlib.readthedocs.io/en/latest/component/data.html",
        }
        _write_json(snapshot / "manifest.json", manifest)
        _write_json(snapshot / "status.json", {"status": "qualified", "manifest_sha256": _sha((snapshot / "manifest.json").read_bytes())})
        return manifest
    except Exception as error:
        _write_json(snapshot / "status.json", {"status": "failed", "error": str(error)})
        raise


def _read_native(provider: Path, calendar: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    from qlib.data import D

    raw = load_qlib_feature_frame(QlibProviderConfig(provider_uri=provider), instruments=["VIX"],
        fields=["$close"], start=START, end=AS_OF)
    native_calendar = pd.DatetimeIndex(D.calendar(freq="day"), name="datetime")
    listed = sorted(D.list_instruments(D.instruments("all"), as_list=True))
    if listed != ["VIX"] or not native_calendar.equals(calendar):
        raise ValueError("blocked: VIX native calendar/instrument audit mismatch")
    values = raw.xs("VIX", level="instrument")["$close"].sort_index().rename("vix_close")
    values.index.name = "datetime"
    if not values.index.equals(calendar) or not np.isfinite(values.to_numpy()).all() or (values <= 0).any():
        raise ValueError("blocked: incomplete VIX native close readback")
    digest = _sha("\n".join(values.index.strftime("%Y-%m-%d")).encode() + values.to_numpy(dtype="<f4").tobytes())
    return values, {"status": "passed", "rows": len(values), "readback_sha256": digest, "storage_dtype": "float32",
                    "source_api": ["qlib.init", "qlib.data.D.calendar", "qlib.data.D.instruments", "qlib.data.D.list_instruments", "qlib.data.D.features"]}


def load_legacy_vix(root: str | Path) -> tuple[pd.Series, dict[str, Any]]:
    """Return fully audited Qlib VIX closes and their immutable source identity.

    This initializes Qlib to the VIX provider; callers must reinitialize their
    execution provider before price backtests.
    """
    root = Path(root).resolve()
    path = root / VIX_SNAPSHOT / "manifest.json"
    raw_manifest = path.read_bytes()
    manifest = json.loads(raw_manifest)
    status = json.loads((root / VIX_SNAPSHOT / "status.json").read_bytes())
    if status.get("status") != "qualified" or status.get("manifest_sha256") != _sha(raw_manifest):
        raise ValueError("blocked: VIX snapshot qualification or manifest hash changed")
    for relative, digest in manifest["file_sha256"].items():
        if _sha((root / relative).read_bytes()) != digest:
            raise ValueError(f"blocked: VIX frozen source/provider hash changed: {relative}")
    calendar, reference = _calendar(root)
    if reference != manifest["reference_calendar"]:
        raise ValueError("blocked: canonical calendar identity changed since VIX freeze")
    values, native = _read_native(root / VIX_PROVIDER, calendar)
    if native != manifest["native_readback"]:
        raise ValueError("blocked: VIX native readback identity changed")
    return values, {"provider_uri": VIX_PROVIDER.as_posix(), "snapshot_manifest": str(VIX_SNAPSHOT / "manifest.json"),
                    "snapshot_manifest_sha256": _sha(raw_manifest), "source_raw_sha256": manifest["raw_sha256"],
                    "source_url": manifest["source_url"], "retrieved_at": manifest["retrieved_at"],
                    "as_of": AS_OF, "reference_calendar": reference, "qualification": manifest["qualification"],
                    "availability": manifest["availability"], "native_readback": native}
