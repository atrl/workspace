from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

import pandas as pd


ANALYSIS_WINDOW_PRESET_SCHEMA_VERSION = "analysis.window_presets.v1"
DEFAULT_ANALYSIS_WINDOW_PRESETS = (
    "ytd",
    "trailing_1m",
    "trailing_3m",
    "trailing_6m",
    "trailing_12m",
    "year_2020",
    "year_2022",
    "period_2023_2024",
)


def _date(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _iso(value: pd.Timestamp) -> str:
    return value.date().isoformat()


def _validate_range(start: pd.Timestamp, end: pd.Timestamp, *, label: str) -> None:
    if end < start:
        raise ValueError(f"{label} end before start: {_iso(start)} > {_iso(end)}")


def _clip_window(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    *,
    available_start: pd.Timestamp,
    available_end: pd.Timestamp,
) -> tuple[str, str] | None:
    clipped_start = max(window_start, available_start)
    clipped_end = min(window_end, available_end)
    if clipped_end < clipped_start:
        return None
    return _iso(clipped_start), _iso(clipped_end)


def _preset_range(name: str, *, end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    if name == "ytd":
        return pd.Timestamp(year=end.year, month=1, day=1), end
    if name == "trailing_1m":
        return end - pd.DateOffset(months=1), end
    if name == "trailing_3m":
        return end - pd.DateOffset(months=3), end
    if name == "trailing_6m":
        return end - pd.DateOffset(months=6), end
    if name == "trailing_12m":
        return end - pd.DateOffset(months=12), end
    if name == "year_2020":
        return pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")
    if name == "year_2022":
        return pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")
    if name == "period_2023_2024":
        return pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")
    raise ValueError(f"unknown analysis window preset: {name}")


def build_analysis_windows(
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    presets: Sequence[str] = DEFAULT_ANALYSIS_WINDOW_PRESETS,
    custom_windows: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, tuple[str, str]]:
    """Build stable fixed analysis windows clipped to the available data range."""

    available_start = _date(start)
    available_end = _date(end)
    _validate_range(available_start, available_end, label="analysis window")

    windows: OrderedDict[str, tuple[str, str]] = OrderedDict()
    for preset in presets:
        window_start, window_end = _preset_range(str(preset), end=available_end)
        clipped = _clip_window(
            window_start,
            window_end,
            available_start=available_start,
            available_end=available_end,
        )
        if clipped is not None:
            windows[str(preset)] = clipped

    for name, (custom_start, custom_end) in (custom_windows or {}).items():
        window_start = _date(custom_start)
        window_end = _date(custom_end)
        _validate_range(window_start, window_end, label=f"custom window '{name}'")
        clipped = _clip_window(
            window_start,
            window_end,
            available_start=available_start,
            available_end=available_end,
        )
        if clipped is not None:
            windows[str(name)] = clipped

    return dict(windows)


def build_analysis_window_table(
    windows: Mapping[str, tuple[str, str]],
    *,
    source: str = "src.analysis.build_analysis_windows",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, (start, end) in windows.items():
        rows.append(
            {
                "schema_version": ANALYSIS_WINDOW_PRESET_SCHEMA_VERSION,
                "window": str(name),
                "start": _iso(_date(start)),
                "end": _iso(_date(end)),
                "source": source,
            }
        )
    return pd.DataFrame(rows, columns=["schema_version", "window", "start", "end", "source"])
