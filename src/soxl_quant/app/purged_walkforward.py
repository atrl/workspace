from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class PurgedWalkforwardSplit:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_rows: int


def _normalise_datetime_frame(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if list(frame.columns).count(date_col) != 1:
        raise ValueError(f"frame must contain exactly one `{date_col}` column")
    out = frame.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)


def build_purged_walkforward_splits(
    frame: pd.DataFrame,
    windows: Mapping[str, tuple[str, str]],
    *,
    date_col: str = "datetime",
    purge_rows: int = 0,
    min_train_rows: int = 1,
    min_test_rows: int = 2,
) -> list[PurgedWalkforwardSplit]:
    """Build expanding-window splits with a row-count purge before each test window."""

    if int(purge_rows) < 0:
        raise ValueError("purge_rows must be non-negative")
    if int(min_train_rows) < 0 or int(min_test_rows) < 0:
        raise ValueError("minimum row counts must be non-negative")

    ordered = _normalise_datetime_frame(frame, date_col)
    dates = ordered[date_col]
    splits: list[PurgedWalkforwardSplit] = []
    for name, (start, end) in windows.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if end_ts < start_ts:
            raise ValueError(f"window `{name}` end is before start")

        test_mask = (dates >= start_ts) & (dates <= end_ts)
        test = ordered.loc[test_mask].copy().reset_index(drop=True)
        if len(test) < int(min_test_rows):
            continue

        first_test_idx = int(test_mask.idxmax())
        train_end_idx = max(first_test_idx - int(purge_rows), 0)
        train = ordered.iloc[:train_end_idx].copy().reset_index(drop=True)
        if len(train) < int(min_train_rows):
            continue

        splits.append(
            PurgedWalkforwardSplit(
                name=str(name),
                train=train,
                test=test,
                train_end=pd.Timestamp(train[date_col].iloc[-1]),
                test_start=pd.Timestamp(test[date_col].iloc[0]),
                test_end=pd.Timestamp(test[date_col].iloc[-1]),
                purge_rows=int(purge_rows),
            )
        )
    return splits


def single_position_frame(
    frame: pd.DataFrame,
    *,
    date_col: str = "datetime",
    position_col: str = "position",
) -> pd.DataFrame:
    """Return an auditable two-column position frame for replay metrics."""

    if list(frame.columns).count(date_col) != 1:
        raise ValueError(f"frame must contain exactly one `{date_col}` column")
    position_count = list(frame.columns).count(position_col)
    if position_count == 0:
        raise ValueError(f"frame missing `{position_col}` column")
    if position_count > 1:
        raise ValueError(f"frame has duplicate `{position_col}` columns")

    out = frame[[date_col, position_col]].copy()
    out = out.rename(columns={date_col: "datetime", position_col: "position"})
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["position"] = out["position"].astype(float).clip(0.0, 1.0)
    return out.dropna(subset=["datetime", "position"]).sort_values("datetime").reset_index(drop=True)


__all__ = ["PurgedWalkforwardSplit", "build_purged_walkforward_splits", "single_position_frame"]
