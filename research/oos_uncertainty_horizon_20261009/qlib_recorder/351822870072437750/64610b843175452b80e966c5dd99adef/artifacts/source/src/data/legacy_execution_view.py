"""Immutable Qlib execution view with one calendar boundary and no new quotes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.data.price_basis import FIELDS, INSTRUMENTS, price_frame_sha256
from src.data.qlib_native import QlibProviderConfig, init_qlib_provider, load_qlib_feature_frame

CANONICAL_PROVIDER = "data/qlib/published_futu_affine_20260925_v1"
EXECUTION_PROVIDER = "data/qlib/legacy_soxl_execution_20260925_v1"
MANIFEST = "execution_boundary_manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(path: Path) -> dict:
    return {str(p.relative_to(path)): _sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def _native(provider: Path):
    from qlib.data import D

    raw = load_qlib_feature_frame(QlibProviderConfig(provider_uri=provider), instruments=list(INSTRUMENTS),
        fields=["$" + name for name in FIELDS], start="2010-03-11", end="2026-09-25").sort_index()
    observed = pd.DatetimeIndex(D.calendar(freq="day", future=False))
    listed = sorted(D.list_instruments(D.instruments("all"), as_list=True))
    if listed != sorted(INSTRUMENTS) or not np.isfinite(raw.to_numpy()).all():
        raise ValueError("blocked: execution-view native universe or observations mismatch")
    return raw, observed


def build_legacy_execution_view(root: str | Path, *, boundary_calendar_provider: str | Path) -> dict:
    """Copy every canonical byte, adding only the next Qlib future-calendar date."""
    from qlib.data import D

    root = Path(root).resolve()
    parent, destination = root / CANONICAL_PROVIDER, root / EXECUTION_PROVIDER
    if destination.exists():
        raise ValueError("blocked: existing execution view is immutable")
    parent_hashes = _files(parent)
    if "calendars/day_future.txt" in parent_hashes:
        raise ValueError("blocked: canonical provider unexpectedly already has a future calendar")
    raw, calendar = _native(parent)
    if len(calendar) != 4162 or str(calendar[0].date()) != "2010-03-11" or str(calendar[-1].date()) != "2026-09-25":
        raise ValueError("blocked: canonical execution range changed")
    native_hash = price_frame_sha256(raw)
    parent_manifest = json.loads((parent / "price_basis_manifest.json").read_text())
    if native_hash != parent_manifest["native_readback"]["frame_sha256"]:
        raise ValueError("blocked: canonical native OHLCV content changed")
    boundary_source = Path(boundary_calendar_provider).resolve()
    init_qlib_provider(QlibProviderConfig(provider_uri=boundary_source))
    future_dates = pd.DatetimeIndex(D.calendar(start_time="2026-09-26", freq="day", future=False))
    if future_dates.empty or str(future_dates[0].date()) != "2026-09-28":
        raise ValueError("blocked: next calendar boundary unavailable or unexpected")
    boundary = str(future_dates[0].date())
    boundary_evidence = {"provider_uri": str(boundary_source),
        "calendar_file_sha256": _sha(boundary_source / "calendars/day.txt"),
        "source_api": "qlib.data.D.calendar(future=False)", "read_start": "2026-09-26",
        "first_returned_date": boundary, "price_features_read_from_source": False}
    shutil.copytree(parent, destination)
    future_path = destination / "calendars/day_future.txt"
    future_path.write_text("\n".join(calendar.strftime("%Y-%m-%d")) + "\n" + boundary + "\n")
    copied_hashes = _files(destination)
    if any(copied_hashes.get(path) != digest for path, digest in parent_hashes.items()) or _files(parent) != parent_hashes:
        raise ValueError("blocked: observed files changed while copying execution view")
    copy_raw, copy_calendar = _native(destination)
    future = pd.DatetimeIndex(D.calendar(freq="day", future=True))
    if not copy_raw.equals(raw) or not copy_calendar.equals(calendar) or len(future) != 4163 or not future[:-1].equals(calendar) or str(future[-1].date()) != boundary:
        raise ValueError("blocked: native execution-view calendar or OHLCV differs")
    manifest = {"schema_version": "legacy_execution_boundary_v1", "status": "qualified",
        "provider_uri": EXECUTION_PROVIDER, "parent_provider_uri": CANONICAL_PROVIDER,
        "as_of": "2026-09-25", "calendar_start": "2010-03-11", "calendar_end": "2026-09-25",
        "boundary_date": boundary, "observed_calendar_rows": len(calendar), "future_calendar_rows": len(future),
        "scope": "One Qlib right interval boundary only; no new observed date, price, return, or execution session.",
        "copied_metadata_scope": "Canonical manifests and qlib.yaml are preserved byte-identically as parent provenance; this manifest identifies the execution view.",
        "parent_file_sha256": parent_hashes, "future_calendar_sha256": _sha(future_path),
        "native_frame_sha256": native_hash, "native_rows": len(raw), "native_observations_equal": True,
        "boundary_calendar_source": boundary_evidence,
        "source_api": ["qlib.init", "qlib.data.D.calendar(future=False)", "qlib.data.D.calendar(future=True)",
                       "qlib.data.D.instruments", "qlib.data.D.list_instruments", "qlib.data.D.features"],
        "qlib_docs": "https://qlib.readthedocs.io/en/latest/reference/api.html#qlib.data.data.CalendarProvider.calendar"}
    (destination / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_legacy_execution_view(root: str | Path) -> tuple[Path, dict]:
    """Revalidate frozen parent bytes and native values, leaving Qlib on the view."""
    from qlib.data import D

    root = Path(root).resolve()
    parent, provider = root / CANONICAL_PROVIDER, root / EXECUTION_PROVIDER
    manifest_path = provider / MANIFEST
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = manifest["parent_file_sha256"]
    if manifest.get("status") != "qualified" or _files(parent) != expected:
        raise ValueError("blocked: execution-view parent identity changed")
    actual = _files(provider)
    if set(actual) != set(expected) | {MANIFEST, "calendars/day_future.txt"}:
        raise ValueError("blocked: execution-view file set changed")
    if any(actual[path] != digest for path, digest in expected.items()):
        raise ValueError("blocked: execution-view observed file bytes differ from canonical source")
    if actual["calendars/day_future.txt"] != manifest["future_calendar_sha256"]:
        raise ValueError("blocked: execution boundary calendar changed")
    parent_raw, parent_calendar = _native(parent)
    raw, observed = _native(provider)
    future = pd.DatetimeIndex(D.calendar(freq="day", future=True))
    if (not raw.equals(parent_raw) or not observed.equals(parent_calendar)
            or len(observed) != manifest["observed_calendar_rows"] or len(future) != len(observed) + 1
            or not future[:-1].equals(observed) or str(future[-1].date()) != manifest["boundary_date"]
            or price_frame_sha256(raw) != manifest["native_frame_sha256"]):
        raise ValueError("blocked: execution-view native readback or boundary differs")
    if manifest_path.read_bytes() != manifest_bytes or _files(parent) != expected:
        raise ValueError("blocked: execution-view source changed during native audit")
    return provider, {"provider_uri": EXECUTION_PROVIDER,
        "execution_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_path": str(Path(EXECUTION_PROVIDER) / MANIFEST),
        "native_frame_sha256": manifest["native_frame_sha256"], "native_rows": len(raw),
        "observed_calendar_rows": len(observed), "future_calendar_rows": len(future),
        "boundary_date": manifest["boundary_date"], "scope": manifest["scope"],
        "parent_file_sha256": expected, "future_calendar_sha256": manifest["future_calendar_sha256"]}
