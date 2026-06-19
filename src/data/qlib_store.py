from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.core import SCHEMA_VERSION, stable_hash

from .qlib_snapshot import CanonicalSnapshot, load_qlib_calendar


@dataclass(frozen=True)
class QlibStoreAudit:
    provider_uri: str
    calendar_start: str
    calendar_end: str
    instruments: dict[str, dict[str, str]]
    missing_instruments: list[str]
    missing_features: dict[str, list[str]]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "provider_uri": self.provider_uri,
            "calendar_start": self.calendar_start,
            "calendar_end": self.calendar_end,
            "instruments": self.instruments,
            "missing_instruments": self.missing_instruments,
            "missing_features": self.missing_features,
            "status": "ok" if not self.missing_instruments and not self.missing_features else "missing_features",
        }


@dataclass(frozen=True)
class DataUpdate:
    instrument: str
    as_of: str
    output_dir: Path
    snapshot_path: Path
    manifest_path: Path
    snapshot_hash: str
    qlib_store: QlibStoreAudit
    row_count: int


def _read_instruments(qlib_dir: Path) -> dict[str, dict[str, str]]:
    instruments_path = qlib_dir / "instruments" / "all.txt"
    if not instruments_path.exists():
        raise FileNotFoundError(f"missing qlib instruments file: {instruments_path}")
    instruments: dict[str, dict[str, str]] = {}
    for line in instruments_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        symbol, start, end = line.split()[:3]
        instruments[symbol] = {"start": start, "end": end}
    return instruments


def inspect_qlib_store(
    qlib_dir: Path,
    *,
    required_instruments: list[str],
    required_fields: list[str],
) -> QlibStoreAudit:
    root = Path(qlib_dir)
    calendar = load_qlib_calendar(root)
    instruments = _read_instruments(root)
    missing_instruments = [symbol for symbol in required_instruments if symbol not in instruments]
    missing_features: dict[str, list[str]] = {}
    for symbol in required_instruments:
        if symbol in missing_instruments:
            continue
        feature_dir = root / "features" / symbol
        missing = [field for field in required_fields if not (feature_dir / f"{field}.day.bin").exists()]
        if missing:
            missing_features[symbol] = missing
    return QlibStoreAudit(
        provider_uri=root.as_posix(),
        calendar_start=calendar.min().date().isoformat(),
        calendar_end=calendar.max().date().isoformat(),
        instruments=instruments,
        missing_instruments=missing_instruments,
        missing_features=missing_features,
    )


def build_data_update_manifest(update: DataUpdate) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "instrument": update.instrument,
        "as_of": update.as_of,
        "row_count": update.row_count,
        "snapshot_path": update.snapshot_path.name,
        "snapshot_hash": update.snapshot_hash,
        "qlib_store": update.qlib_store.to_manifest(),
    }


def write_canonical_snapshot_update(
    snapshot: CanonicalSnapshot,
    *,
    output_dir: Path,
    qlib_dir: Path,
) -> DataUpdate:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = target_dir / f"{snapshot.instrument}_{snapshot.as_of}_snapshot.csv"
    data = snapshot.data.copy()
    data.to_csv(snapshot_path, index=False)
    snapshot_hash = stable_hash(data.to_dict("records"))
    qlib_store = inspect_qlib_store(
        Path(qlib_dir),
        required_instruments=[snapshot.instrument],
        required_fields=[column for column in data.columns if column != "date"],
    )
    update = DataUpdate(
        instrument=snapshot.instrument,
        as_of=snapshot.as_of,
        output_dir=target_dir,
        snapshot_path=snapshot_path,
        manifest_path=target_dir / "manifest.json",
        snapshot_hash=snapshot_hash,
        qlib_store=qlib_store,
        row_count=len(data),
    )
    update.manifest_path.write_text(json.dumps(build_data_update_manifest(update), indent=2, sort_keys=True), encoding="utf-8")
    return update
