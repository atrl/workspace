"""Generic horizon label + clock + dataset contracts (default 5 preserves the H5 behavior).

New matched-horizon work (H20) reuses this instead of cloning the H5 module. The default
horizon is 5 and reproduces the registered H5 label formula; H20 callers pass
``horizon_sessions=20``. Units, clocks, maturity and the calendar source are explicit so
the horizon, the mean/variance units and the daily risk unit are never mixed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.convex_allocation import HORIZON_SESSIONS as LEGACY_HORIZON_SESSIONS

DEFAULT_HORIZON_SESSIONS = 5
DAILY_RISK_UNIT = "one_session_simple_return_squared"
NY = "America/New_York"
ALLOWED_INDEX_ZONES = (None, "America/New_York")

_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def number_word(value: int) -> str:
    value = int(value)
    if value < 20:
        return _ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]}_{_ONES[ones]}"
    return str(value)


def strict_positive_int(value, *, name: str = "horizon_sessions") -> int:
    """Accept only a positive integer; reject bool, fractional/non-finite, zero and negatives."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer, float, np.floating)):
        raise ValueError(f"blocked: {name} must be a positive integer")
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or float(value) != int(value):
            raise ValueError(f"blocked: {name} must be a positive integer (no fractional horizon)")
    result = int(value)
    if result < 1:
        raise ValueError(f"blocked: {name} must be a positive integer")
    return result


def mean_unit(horizon_sessions: int) -> str:
    return f"{number_word(horizon_sessions)}_session_simple_return"


def mean_variance_unit(horizon_sessions: int) -> str:
    return f"{number_word(horizon_sessions)}_session_simple_return_squared"


def unit_tags(horizon_sessions: int = DEFAULT_HORIZON_SESSIONS) -> dict:
    """Explicit unit contract for a horizon; arrays must carry these tags to be accepted."""
    horizon_sessions = strict_positive_int(horizon_sessions)
    return {
        "horizon_sessions": horizon_sessions,
        "mean_unit": mean_unit(horizon_sessions),
        "mean_variance_unit": mean_variance_unit(horizon_sessions),
        "daily_risk_unit": DAILY_RISK_UNIT,
    }


def _validate_signal_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("blocked: signal index must be a DatetimeIndex")
    if index.hasnans:
        raise ValueError("blocked: signal index contains NaT")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("blocked: signal index must be unique and increasing")
    zone = str(index.tz) if index.tz is not None else None
    if zone not in ALLOWED_INDEX_ZONES:
        raise ValueError(f"blocked: signal index timezone {zone} not allowed")
    if not ((index.hour == 0) & (index.minute == 0) & (index.second == 0)
            & (index.microsecond == 0) & (index.nanosecond == 0)).all():
        raise ValueError("blocked: signal index must be midnight-normalized session dates")
    return index


def _session_dates(values, *, name: str = "session dates") -> pd.DatetimeIndex:
    """Unified session-date contract: naive calendar dates or America/New_York midnight.

    tz-aware inputs must already be New York session dates; other zones (including UTC)
    are rejected rather than converted, because a UTC->NY conversion can shift the date.
    """
    index = pd.DatetimeIndex(pd.to_datetime(values))
    if index.hasnans:
        raise ValueError(f"blocked: {name} contains NaT")
    if index.tz is not None:
        if str(index.tz) != NY:
            raise ValueError(f"blocked: {name} tz-aware input must be America/New_York session dates")
        index = index.tz_convert(NY)
    if not ((index.hour == 0) & (index.minute == 0) & (index.second == 0)
            & (index.microsecond == 0) & (index.nanosecond == 0)).all():
        raise ValueError(f"blocked: {name} must be midnight session dates")
    return index


def predictor_clock(signal_dates) -> pd.DatetimeIndex:
    """Predictor decision clock: signal session date 16:15 America/New_York (UTC)."""
    stamps = _session_dates(signal_dates, name="signal dates")
    if stamps.tz is None:
        stamps = stamps.tz_localize(NY)
    return (stamps + pd.Timedelta(hours=16, minutes=15)).tz_convert("UTC")


def allocation_clock(trade_dates) -> pd.DatetimeIndex:
    """Allocation clock: trade session open captured after 09:31 America/New_York (UTC)."""
    stamps = _session_dates(trade_dates, name="trade dates")
    if stamps.tz is None:
        stamps = stamps.tz_localize(NY)
    return (stamps + pd.Timedelta(hours=9, minutes=31)).tz_convert("UTC")


def assert_pretrade_not_relabeled(allocation_at, claimed_at) -> None:
    """Reject claiming a next-open pretrade state as known at the prior signal clock.

    Raises when any claimed time is earlier than the actual allocation observation time;
    NaT in either clock is rejected rather than silently passing.
    """
    alloc = pd.DatetimeIndex(pd.to_datetime(allocation_at, utc=True))
    claim = pd.DatetimeIndex(pd.to_datetime(claimed_at, utc=True))
    if alloc.hasnans or claim.hasnans:
        raise ValueError("blocked: clock contains NaT")
    if len(alloc) != len(claim):
        raise ValueError("blocked: allocation/prediction clock length mismatch")
    if bool((alloc > claim).any()):
        raise ValueError("blocked: next-open pretrade state cannot be claimed known at prior signal 16:15")


def _open_at(days: pd.Series, hour: int, minute: int) -> pd.Series:
    # allow NaT for terminal-unknown exits; only real session dates get a clock
    values = pd.DatetimeIndex(pd.to_datetime(days))
    out = pd.Series(pd.NaT, index=days.index, dtype="datetime64[ns, UTC]")
    mask = ~values.isna()
    if mask.any():
        sub = values[mask]
        if sub.tz is None:
            sub = sub.tz_localize(NY)
        elif str(sub.tz) != NY:
            raise ValueError("blocked: label exit dates tz-aware input must be America/New_York session dates")
        if not ((sub.hour == 0) & (sub.minute == 0) & (sub.second == 0)).all():
            raise ValueError("blocked: label exit dates must be midnight session dates")
        stamps = (sub + pd.Timedelta(hours=hour, minutes=minute)).tz_convert("UTC")
        out.iloc[np.where(mask)[0]] = pd.DatetimeIndex(stamps)
    return out


def validate_price_calendar(calendar, prices: pd.DataFrame) -> pd.DatetimeIndex:
    """The price frame must be 1:1 with the actual D.calendar (no missing/extra/duplicate day)."""
    cal = pd.DatetimeIndex(calendar)
    if cal.hasnans or cal.has_duplicates or not cal.is_monotonic_increasing:
        raise ValueError("blocked: calendar must be unique, ordered and non-NaT")
    if not pd.DatetimeIndex(prices.index).equals(cal):
        raise ValueError("blocked: price frame must be 1:1 with the actual D.calendar (missing/extra day)")
    if prices.isna().any().any():
        raise ValueError("blocked: missing prices in the calendar-aligned frame")
    return cal


def build_horizon_labels(frame: pd.DataFrame, execution_prices: pd.DataFrame, *,
                         horizon_sessions: int = DEFAULT_HORIZON_SESSIONS,
                         availability_hour: int = 9, availability_minute: int = 30,
                         symbols=("SOXL", "SOXX"), calendar=None) -> tuple[pd.DataFrame, dict]:
    """Attach horizon labels open[t+h+1]/open[t+1]-1 from a calendar-aligned price frame.

    ``calendar`` is the authoritative calendar (e.g. the real D.calendar); pass it to
    verify the price frame is 1:1 with no missing/extra day so t+21 cannot be compressed.
    When omitted, ``execution_prices.index`` is used as the provided calendar.
    """
    horizon_sessions = strict_positive_int(horizon_sessions)
    signal = _validate_signal_index(frame)
    source = "explicit calendar" if calendar is not None else "provided 1:1 calendar"
    calendar_index = execution_prices.index if calendar is None else calendar
    calendar = validate_price_calendar(calendar_index, execution_prices)
    if not signal.isin(calendar).all():
        raise ValueError("blocked: signal dates must be present in the actual calendar")
    for symbol in symbols:
        column = f"open_{symbol}"
        if column not in execution_prices:
            raise ValueError(f"blocked: missing {column} for horizon labels")
        opening = execution_prices[column].astype("float64")
        if not np.isfinite(opening.to_numpy()).all() or (opening <= 0).any():
            raise ValueError(f"blocked: invalid opening prices in {column}")

    dates = pd.Series(calendar, index=calendar)
    entry = dates.shift(-1).reindex(signal)
    exit_day = dates.shift(-(horizon_sessions + 1)).reindex(signal)
    result = frame.copy()
    if "trade_date" in result.columns:
        if not pd.to_datetime(result["trade_date"]).reset_index(drop=True).equals(
                pd.to_datetime(entry).reset_index(drop=True)):
            raise ValueError("blocked: trade_date does not match calendar t+1 entry (wrong or missing day)")
    result["trade_date"] = entry.to_numpy()
    result[f"exit_date_h{horizon_sessions}"] = exit_day.to_numpy()
    for symbol in symbols:
        opening = execution_prices[f"open_{symbol}"].astype("float64")
        label = (opening.shift(-(horizon_sessions + 1)) / opening.shift(-1) - 1).reindex(signal)
        result[f"{symbol.lower()}_return_h{horizon_sessions}"] = label.to_numpy()
    result[f"label_h{horizon_sessions}_available_at"] = _open_at(exit_day, availability_hour, availability_minute).to_numpy()
    label_columns = [f"{s.lower()}_return_h{horizon_sessions}" for s in symbols]
    unknown = result[label_columns].isna().any(axis=1)
    audit = {
        "horizon_sessions": horizon_sessions,
        "formula": f"open[t+{horizon_sessions + 1}]/open[t+1]-1",
        "mean_unit": mean_unit(horizon_sessions),
        "mean_variance_unit": mean_variance_unit(horizon_sessions),
        "daily_risk_unit": DAILY_RISK_UNIT,
        "availability_clock": f"{availability_hour:02d}:{availability_minute:02d} America/New_York",
        "calendar_source": source,
        "terminal_unknown_labels_retained": int(unknown.sum()),
        "terminal_unknown_required_columns": label_columns + [f"label_h{horizon_sessions}_available_at"],
        "dropna_or_date_deletion": False,
        "rows": int(len(result)),
        "calendar_rows": int(len(calendar)),
    }
    return result, audit


def actual_calendar_prices(provider_uri, *, start_time, end_time, symbols, freq: str = "day"):
    """Read the actual Qlib ``D.calendar`` and ``$open`` features (read-only)."""
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=str(Path(provider_uri).resolve()), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(start_time=start_time, end_time=end_time, freq=freq))
    raw = D.features(list(symbols), ["$open"], start_time=start_time, end_time=end_time, freq=freq)["$open"]
    raw = raw.unstack("instrument")
    return calendar, raw


def build_horizon_labels_from_qlib(frame: pd.DataFrame, *, provider_uri, start_time, end_time,
                                   symbols=("SOXL", "SOXX"), horizon_sessions: int = DEFAULT_HORIZON_SESSIONS,
                                   availability_hour: int = 9, availability_minute: int = 31,
                                   freq: str = "day") -> tuple[pd.DataFrame, dict]:
    """Actual-source wrapper: read the real D.calendar, verify 1:1 prices, then build labels."""
    calendar, raw = actual_calendar_prices(provider_uri, start_time=start_time, end_time=end_time,
                                           symbols=symbols, freq=freq)
    validate_price_calendar(calendar, raw)
    prices = pd.DataFrame({f"open_{symbol}": raw[symbol].to_numpy(dtype="float64") for symbol in symbols}, index=calendar)
    labels, audit = build_horizon_labels(frame, prices, horizon_sessions=horizon_sessions,
                                         availability_hour=availability_hour, availability_minute=availability_minute,
                                         symbols=symbols, calendar=calendar)
    audit["calendar_source"] = "actual D.calendar"
    audit["calendar_sha256_rows"] = int(len(calendar))
    return labels, audit


def mature_mask(available_at, cutoff, *, naive_timezone=None) -> pd.Series:
    """Strict ``label_available_at < cutoff`` preserving the input index; NaT is not mature.

    A naive ``cutoff`` requires an explicit ``naive_timezone`` rather than being assumed.
    """
    series = available_at if isinstance(available_at, pd.Series) else pd.Series(list(available_at))
    available = pd.to_datetime(series, utc=True)
    cut = pd.Timestamp(cutoff)
    if cut.tzinfo is None:
        if naive_timezone is None:
            raise ValueError("blocked: naive cutoff requires an explicit timezone")
        cut = cut.tz_localize(naive_timezone)
    cut = cut.tz_convert("UTC")
    result = available.lt(cut)
    result = result.where(available.notna(), False).astype(bool)
    result.index = series.index
    return result


def attach_horizon_scope(dataset, horizon_sessions: int = DEFAULT_HORIZON_SESSIONS):
    """Tag a DatasetH with its horizon so a wrong-horizon dataset is blocked downstream."""
    horizon_sessions = strict_positive_int(horizon_sessions)
    setattr(dataset, "horizon_sessions", horizon_sessions)
    return dataset


def require_horizon_scope(dataset, horizon_sessions: int) -> None:
    """Guard: the DatasetH must carry exactly this horizon (e.g. 20 for H20, 5 for H5)."""
    horizon_sessions = strict_positive_int(horizon_sessions)
    if getattr(dataset, "horizon_sessions", None) != horizon_sessions:
        raise ValueError("blocked: horizon DatasetH does not match the required horizon")
