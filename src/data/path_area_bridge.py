"""Fixed signed path-area inputs for the P2 local-action bridge.

This module is deliberately a pure transform. It accepts already admitted
Qlib-native minute marks and coverage metadata; it has no provider or source
loading path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


AREA_COLUMN = "path_area_signed"
CONTROL_COLUMNS = (
    "path_control_X",
    "path_control_Y",
    "path_control_X_squared",
    "path_control_XY",
    "path_control_Y_squared",
    "path_control_Qx",
    "path_control_Qxy",
    "path_control_Qy",
)
PATH_COLUMNS = (AREA_COLUMN, *CONTROL_COLUMNS)
SYMBOLS = ("SOXL", "SOXX")
PREFIX_START = pd.Timedelta(hours=9, minutes=31)
PREFIX_CLOSES = 15


def signed_path_area(points: np.ndarray) -> tuple[float, dict[str, float]]:
    """Compute the frozen SOXX-x/SOXL-y area and eight fixed controls.

    ``points`` contains P0 followed by the fifteen close marks. Columns are
    physical raw-NONE prices in (SOXL, SOXX) order. Native float32 values are
    widened before division; accumulation is a specified left fold.
    """
    raw = np.asarray(points)
    if raw.shape != (PREFIX_CLOSES + 1, 2):
        raise ValueError("blocked: path must contain P0 and fifteen two-asset closes")
    prices = raw.astype(np.float64, copy=False)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("blocked: declared-valid path contains nonfinite or nonpositive prices")
    soxl0, soxx0 = prices[0, 0], prices[0, 1]
    x = prices[:, 1] / soxx0 - 1.0
    y = prices[:, 0] / soxl0 - 1.0
    area_twice = qx = qxy = qy = 0.0
    for index in range(1, len(x)):
        dx = float(x[index] - x[index - 1])
        dy = float(y[index] - y[index - 1])
        area_twice += float(x[index - 1] * dy - y[index - 1] * dx)
        qx += dx * dx
        qxy += dx * dy
        qy += dy * dy
    controls = {
        CONTROL_COLUMNS[0]: float(x[-1]),
        CONTROL_COLUMNS[1]: float(y[-1]),
        CONTROL_COLUMNS[2]: float(x[-1] * x[-1]),
        CONTROL_COLUMNS[3]: float(x[-1] * y[-1]),
        CONTROL_COLUMNS[4]: float(y[-1] * y[-1]),
        CONTROL_COLUMNS[5]: qx,
        CONTROL_COLUMNS[6]: qxy,
        CONTROL_COLUMNS[7]: qy,
    }
    area = float(0.5 * area_twice)
    if not np.isfinite([area, *controls.values()]).all():
        raise ValueError("blocked: fixed path arithmetic produced a nonfinite value")
    return area, controls


def _feature_frame(minute_marks: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(minute_marks, pd.DataFrame):
        raise TypeError("minute marks must be a DataFrame")
    if isinstance(minute_marks.columns, pd.MultiIndex):
        if minute_marks.columns.nlevels != 2:
            raise ValueError("blocked: minute mark column groups differ")
        if "feature" not in minute_marks.columns.get_level_values(0):
            raise ValueError("blocked: Qlib minute marks have no feature group")
        frame = minute_marks["feature"].copy()
    else:
        frame = minute_marks.copy()
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise ValueError("blocked: minute marks need a datetime/instrument index")
    if list(frame.index.names) != ["datetime", "instrument"]:
        raise ValueError("blocked: minute mark index levels differ")
    required = {"$open", "$close", "$price_valid"}
    if not required.issubset(frame.columns):
        raise ValueError("blocked: admitted minute mark fields are incomplete")
    if frame.index.has_duplicates:
        raise ValueError("blocked: duplicate native minute mark")
    return frame.sort_index()


def _signal_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        if index.nlevels != 2 or "datetime" not in index.names:
            raise ValueError("blocked: source Dataset index is not datetime/instrument")
        values = index.get_level_values("datetime")
    else:
        values = index
    result = pd.DatetimeIndex(values)
    if result.tz is not None:
        result = result.tz_localize(None)
    if result.has_duplicates:
        raise ValueError("blocked: duplicate signal-date timing rows")
    return result


def build_path_area_inputs(
    minute_marks: pd.DataFrame,
    timing: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    prefix_mask_column: str = "prefix_mask",
) -> pd.DataFrame:
    """Build one fixed 09:30--09:45 path input per admitted timing row.

    Rows with a missing prefix retain NaN for the canonical area and controls
    and receive mask zero. A prefix that is declared complete but has malformed
    required marks raises a mechanical error instead of becoming unavailable.
    """
    marks = _feature_frame(minute_marks)
    if not isinstance(timing, pd.DataFrame) or not {"trade_date"}.issubset(timing.columns):
        raise ValueError("blocked: timing must provide actual trade_date")
    if not isinstance(coverage, pd.DataFrame) or prefix_mask_column not in coverage:
        raise ValueError("blocked: P1 prefix coverage mask is missing")
    signal_dates = _signal_index(timing.index)
    coverage_dates = _signal_index(coverage.index)
    if len(signal_dates) != len(timing) or len(coverage_dates) != len(coverage):
        raise ValueError("blocked: timing/coverage index mismatch")
    timing = timing.copy()
    timing.index = signal_dates
    coverage = coverage.copy()
    coverage.index = coverage_dates
    if timing.index.has_duplicates or coverage.index.has_duplicates:
        raise ValueError("blocked: duplicate timing or coverage dates")
    if not timing.index.equals(coverage.index):
        raise ValueError("blocked: timing and P1 prefix coverage calendars differ")

    output = []
    for signal_date, row in timing.iterrows():
        trade_date = pd.Timestamp(row["trade_date"])
        if trade_date.tzinfo is not None:
            trade_date = trade_date.tz_localize(None)
        if trade_date != trade_date.normalize():
            raise ValueError("blocked: trade_date must be a native session date")
        raw_mask = coverage.at[signal_date, prefix_mask_column]
        if pd.isna(raw_mask) or raw_mask not in (True, False, 0, 1):
            raise ValueError("blocked: P1 prefix mask must be a known binary value")
        prefix_ok = bool(raw_mask)
        values = {column: np.nan for column in PATH_COLUMNS}
        mask = int(prefix_ok)
        if prefix_ok:
            start = trade_date + PREFIX_START
            stamps = pd.date_range(start, periods=PREFIX_CLOSES, freq="min")
            path = np.empty((PREFIX_CLOSES + 1, 2), dtype=np.float64)
            for symbol_index, symbol in enumerate(SYMBOLS):
                for stamp in stamps:
                    key = (stamp, symbol)
                    if key not in marks.index:
                        raise ValueError("blocked: P1 prefix mask declares a missing native bar")
                    mark = marks.loc[key]
                    if int(mark["$price_valid"]) != 1:
                        raise ValueError("blocked: P1 prefix mask declares an invalid native bar")
                first = marks.loc[(stamps[0], symbol)]
                path[0, symbol_index] = np.float64(first["$open"])
                path[1:, symbol_index] = marks.loc[
                    [(stamp, symbol) for stamp in stamps], "$close"
                ].to_numpy(dtype=np.float64)
            area, controls = signed_path_area(path)
            values = {AREA_COLUMN: area, **controls}
        output.append({"signal_date": signal_date, "trade_date": trade_date,
                       **values, "prefix_mask": mask})
    result = pd.DataFrame(output).set_index("signal_date").sort_index()
    if not result.index.equals(timing.index.sort_values()):
        raise ValueError("blocked: output path index differs from source calendar")
    return result


def route_prediction_rows(timing: pd.DataFrame) -> dict[int, pd.DatetimeIndex]:
    """Map signal dates to actual trade years; preserve year-boundary rows."""
    if "trade_date" not in timing:
        raise ValueError("blocked: timing lacks trade_date")
    signal = _signal_index(timing.index)
    dates = pd.to_datetime(timing["trade_date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    result = {}
    for year in range(2018, 2023):
        result[year] = pd.DatetimeIndex(signal[dates.dt.year.to_numpy() == year])
    return result


def build_raw_training_point_prices(
    minute_marks: pd.DataFrame,
    session_schedule: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Project admitted raw marks onto the deterministic physical-account grid.

    Qlib timestamps are policy/execution times. The open consumed at 09:47 is
    read from the right-edge 09:48 bar's ``$open``; the 10:32 mark is read from
    10:33 ``$open``. Session-close marks use the exchange schedule.
    """
    marks = _feature_frame(minute_marks)
    if not isinstance(session_schedule, pd.DataFrame) or not {"open", "close"}.issubset(session_schedule):
        raise ValueError("blocked: XNYS session schedule requires open and close")
    schedule = session_schedule.copy()
    schedule.index = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    if schedule.index.has_duplicates:
        raise ValueError("blocked: duplicate scheduled session")
    lower, upper = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    schedule = schedule.loc[(schedule.index >= lower) & (schedule.index <= upper)]
    days = schedule.index
    if not len(days) or days[0] != lower or days[-1] != upper:
        raise ValueError("needs_data: schedule does not cover the full requested account span")
    rows = []
    for day, session in schedule.iterrows():
        opening, closing = pd.Timestamp(session["open"]), pd.Timestamp(session["close"])
        if opening.tzinfo is None or closing.tzinfo is None:
            raise ValueError("blocked: XNYS schedule timestamps must carry a timezone")
        opening = opening.tz_convert("America/New_York")
        closing = closing.tz_convert("America/New_York")
        if opening != day.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30):
            raise ValueError("blocked: scheduled regular-session open differs from P2 clock")
        close_stamp = closing.tz_localize(None)
        source_marks = (
            (day + pd.Timedelta(hours=9, minutes=45), day + pd.Timedelta(hours=9, minutes=45), "$close"),
            (day + pd.Timedelta(hours=9, minutes=47), day + pd.Timedelta(hours=9, minutes=48), "$open"),
            (day + pd.Timedelta(hours=10, minutes=32), day + pd.Timedelta(hours=10, minutes=33), "$open"),
            (close_stamp, close_stamp, "$close"),
        )
        for policy_stamp, source_stamp, field in source_marks:
            values = {}
            for symbol in SYMBOLS:
                key = (source_stamp, symbol)
                if key not in marks.index:
                    raise ValueError("needs_data: required raw physical-account mark is missing")
                source = marks.loc[key]
                value = source[field]
                if int(source["$price_valid"]) != 1 or not np.isfinite(value) or float(value) <= 0:
                    raise ValueError("needs_data: required raw physical-account mark is invalid")
                values[symbol] = float(np.float64(value))
            rows.append({"stamp": policy_stamp, **values})
    result = pd.DataFrame(rows).set_index("stamp").sort_index()
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError("blocked: physical point grid is not unique and chronological")
    return result.loc[:, list(SYMBOLS)]
