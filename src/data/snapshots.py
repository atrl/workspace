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
