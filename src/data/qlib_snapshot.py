from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CanonicalSnapshot:
    instrument: str
    as_of: str
    data: pd.DataFrame
    attrs: dict[str, Any]


def load_qlib_calendar(qlib_dir: Path) -> pd.DatetimeIndex:
    calendar_path = Path(qlib_dir) / "calendars" / "day.txt"
    if not calendar_path.exists():
        raise FileNotFoundError(f"missing qlib calendar: {calendar_path}")
    dates = [line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DatetimeIndex(pd.to_datetime(dates), name="date")


def canonical_snapshot_from_panel(
    panel: pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex,
    instrument: str,
    as_of: str,
    source_path: Path,
) -> CanonicalSnapshot:
    if "date" not in panel.columns:
        raise ValueError("panel must include a date column")
    as_of_ts = pd.Timestamp(as_of)
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    valid_dates = calendar[calendar <= as_of_ts]
    snapshot = frame.loc[frame["date"].isin(valid_dates)].sort_values("date").reset_index(drop=True)
    attrs = {
        "data_source": "qlib_calendar_plus_canonical_snapshot",
        "qlib_calendar_size": len(calendar),
        "source_path": Path(source_path).as_posix(),
    }
    return CanonicalSnapshot(instrument=instrument, as_of=as_of_ts.date().isoformat(), data=snapshot, attrs=attrs)
