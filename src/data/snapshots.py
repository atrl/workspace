from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.core import DatasetSnapshot
from src.core.contracts import stable_hash


CANONICAL_FRAME_SCHEMA_VERSION = "data.canonical_frame.v1"
CANONICAL_SNAPSHOT_EXPORT_SCHEMA_VERSION = "data.canonical_snapshot.export.v1"
DATA_UPDATE_MANIFEST_SCHEMA_VERSION = "data.update_manifest.v1"
DATA_REGISTRY_SCHEMA_VERSION = "data.registry.v1"


@dataclass(frozen=True)
class CanonicalSnapshot:
    frame: pd.DataFrame
    dataset: DatasetSnapshot


def _date_string(value: pd.Timestamp) -> str:
    return value.date().isoformat()


def _frame_schema_hash(frame: pd.DataFrame) -> str:
    payload = {
        "schema_version": CANONICAL_FRAME_SCHEMA_VERSION,
        "columns": [(str(column), str(dtype)) for column, dtype in zip(frame.columns, frame.dtypes, strict=True)],
    }
    return stable_hash(payload)


def _prepare_calendar(calendar: pd.DatetimeIndex | list[str] | tuple[str, ...]) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(pd.to_datetime(calendar), name="date").normalize()
    return values.drop_duplicates().sort_values()


def _prepare_frame(frame: pd.DataFrame, *, date_column: str) -> pd.DataFrame:
    if date_column not in frame.columns:
        raise ValueError(f"canonical frame missing date column: {date_column}")
    result = frame.copy()
    result[date_column] = pd.to_datetime(result[date_column]).dt.normalize()
    sort_columns = [date_column]
    if "instrument" in result.columns:
        sort_columns.append("instrument")
    return result.sort_values(sort_columns).reset_index(drop=True)


def _validate_no_future_rows(frame: pd.DataFrame, *, date_column: str, as_of: str) -> None:
    as_of_ts = pd.Timestamp(as_of).normalize()
    future = frame.loc[frame[date_column] > as_of_ts]
    if not future.empty:
        first = _date_string(pd.Timestamp(future[date_column].min()))
        raise ValueError(f"future rows after as_of={as_of}: first={first}")


def _validate_unique_keys(frame: pd.DataFrame, *, date_column: str) -> None:
    key_columns = [date_column]
    if "instrument" in frame.columns:
        key_columns.append("instrument")
    duplicated = frame.duplicated(key_columns)
    if duplicated.any():
        raise ValueError(f"duplicate canonical keys: {key_columns}")


def build_canonical_snapshot(
    frame: pd.DataFrame,
    *,
    name: str,
    universe: tuple[str, ...],
    calendar: pd.DatetimeIndex | list[str] | tuple[str, ...],
    calendar_path: str,
    source_paths: tuple[str, ...],
    start: str,
    end: str,
    as_of: str,
    date_column: str = "date",
) -> CanonicalSnapshot:
    prepared = _prepare_frame(frame, date_column=date_column)
    _validate_no_future_rows(prepared, date_column=date_column, as_of=as_of)
    _validate_unique_keys(prepared, date_column=date_column)

    calendar_values = _prepare_calendar(calendar)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    window_calendar = calendar_values[(calendar_values >= start_ts) & (calendar_values <= end_ts)]
    window_frame = prepared.loc[(prepared[date_column] >= start_ts) & (prepared[date_column] <= end_ts)].copy()
    aligned = window_frame.loc[window_frame[date_column].isin(window_calendar)].copy()
    dropped = int(len(window_frame) - len(aligned))
    if aligned.empty:
        raise ValueError(f"no canonical rows between {start} and {end}")

    aligned = aligned.sort_values([date_column, "instrument"] if "instrument" in aligned.columns else [date_column])
    aligned = aligned.reset_index(drop=True)
    metadata: dict[str, Any] = {
        "canonical_frame_schema_version": CANONICAL_FRAME_SCHEMA_VERSION,
        "frame_schema_hash": _frame_schema_hash(aligned),
        "row_count": int(len(aligned)),
        "column_count": int(len(aligned.columns)),
        "calendar_row_count": int(len(window_calendar)),
        "dropped_non_calendar_rows": dropped,
        "as_of": pd.Timestamp(as_of).date().isoformat(),
        "date_column": date_column,
        "source_paths": source_paths,
    }
    dataset = DatasetSnapshot(
        name=name,
        universe=universe,
        start=_date_string(pd.Timestamp(aligned[date_column].min())),
        end=_date_string(pd.Timestamp(aligned[date_column].max())),
        calendar_path=calendar_path,
        data_paths=source_paths,
        metadata=metadata,
    )
    return CanonicalSnapshot(frame=aligned, dataset=dataset)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_canonical_snapshot(
    snapshot: CanonicalSnapshot,
    output_dir: str | Path,
    *,
    frame_filename: str = "canonical_frame.csv",
    snapshot_filename: str = "dataset_snapshot.json",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame_path = output / frame_filename
    snapshot_path = output / snapshot_filename

    frame = snapshot.frame.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    frame.to_csv(frame_path, index=False)

    payload = {
        "schema_version": CANONICAL_SNAPSHOT_EXPORT_SCHEMA_VERSION,
        "dataset": snapshot.dataset.to_dict(),
        "dataset_schema_hash": snapshot.dataset.schema_hash(),
        "paths": {
            "frame": frame_filename,
            "dataset_snapshot": snapshot_filename,
        },
    }
    snapshot_path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": DATA_REGISTRY_SCHEMA_VERSION, "datasets": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("schema_version", DATA_REGISTRY_SCHEMA_VERSION)
    payload.setdefault("datasets", {})
    return payload


def _registry_record(*, dataset_key: str, manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    dataset = manifest["dataset"]
    metadata = dataset.get("metadata", {})
    return {
        "dataset_key": dataset_key,
        "dataset_name": dataset.get("name", ""),
        "snapshot_path": str(output_dir),
        "manifest_path": str(output_dir / manifest["paths"]["manifest"]),
        "manifest_hash": manifest["manifest_hash"],
        "dataset_schema_hash": manifest["dataset_schema_hash"],
        "start": dataset.get("start", ""),
        "end": dataset.get("end", ""),
        "as_of": metadata.get("as_of", ""),
        "row_count": int(metadata.get("row_count", 0) or 0),
        "frame_schema_hash": metadata.get("frame_schema_hash", ""),
        "published_at": manifest.get("published_at", ""),
    }


def _write_registry(
    *,
    registry_path: Path,
    dataset_key: str,
    manifest: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    registry = _load_registry(registry_path)
    datasets = registry.setdefault("datasets", {})
    record = _registry_record(dataset_key=dataset_key, manifest=manifest, output_dir=output_dir)
    entry = datasets.setdefault(dataset_key, {"history": []})
    history = list(entry.get("history", []))
    if all(item.get("manifest_hash") != record["manifest_hash"] for item in history):
        history.append(record)
    entry["current"] = record
    entry["history"] = history
    datasets[dataset_key] = entry
    registry["registry_hash"] = stable_hash(
        {
            "schema_version": registry["schema_version"],
            "datasets": datasets,
        }
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(_json_ready(registry), ensure_ascii=False, indent=2), encoding="utf-8")
    return registry


def publish_canonical_snapshot(
    snapshot: CanonicalSnapshot,
    output_dir: str | Path,
    *,
    dataset_key: str,
    registry_path: str | Path | None = None,
    published_at: str | None = None,
    frame_filename: str = "canonical_frame.csv",
    snapshot_filename: str = "dataset_snapshot.json",
    manifest_filename: str = "data_update_manifest.json",
) -> dict[str, Any]:
    """Publish a canonical snapshot and optionally update the data registry current pointer."""

    output = Path(output_dir)
    export = write_canonical_snapshot(
        snapshot,
        output,
        frame_filename=frame_filename,
        snapshot_filename=snapshot_filename,
    )
    dataset_payload = snapshot.dataset.to_dict()
    publish_time = published_at or f"{dataset_payload.get('end', '')}T00:00:00Z"
    manifest_without_hash = {
        "schema_version": DATA_UPDATE_MANIFEST_SCHEMA_VERSION,
        "status": "published",
        "dataset_key": dataset_key,
        "published_at": publish_time,
        "dataset": dataset_payload,
        "dataset_schema_hash": snapshot.dataset.schema_hash(),
        "canonical_export_schema_version": export["schema_version"],
        "paths": {
            "frame": frame_filename,
            "dataset_snapshot": snapshot_filename,
            "manifest": manifest_filename,
        },
    }
    manifest = {
        **manifest_without_hash,
        "manifest_hash": stable_hash(manifest_without_hash),
    }
    (output / manifest_filename).write_text(json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    if registry_path is not None:
        _write_registry(
            registry_path=Path(registry_path),
            dataset_key=dataset_key,
            manifest=manifest,
            output_dir=output,
        )
    return _json_ready(manifest)
