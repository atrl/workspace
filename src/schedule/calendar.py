from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable, Mapping


MARKET_CALENDAR_DECISION_SCHEMA_VERSION = "schedule.market_calendar_decision.v1"


def _day(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("date value is required")
    return date.fromisoformat(text[:10])


def _day_set(values: Iterable[object]) -> set[date]:
    return {_day(value) for value in values}


def _day_string(value: date | None) -> str:
    return value.isoformat() if value is not None else ""


def classify_market_calendar_day(
    as_of: object,
    *,
    trading_calendar: Iterable[object],
    half_days: Iterable[object] = (),
    holiday_reasons: Mapping[str, str] | None = None,
    allow_half_day: bool = True,
) -> dict[str, Any]:
    """Classify a scheduled run date before data/model/dashboard/paper jobs execute."""

    current = _day(as_of)
    trading_days = sorted(_day_set(trading_calendar))
    if not trading_days:
        raise ValueError("trading_calendar must contain at least one date")
    half_day_set = _day_set(half_days)
    reasons = {str(key)[:10]: str(value) for key, value in dict(holiday_reasons or {}).items()}
    previous_trading_date = next((day for day in reversed(trading_days) if day < current), None)
    next_future_trading_date = next((day for day in trading_days if day > current), None)

    if current in trading_days:
        is_half_day = current in half_day_set
        should_run = bool(allow_half_day or not is_half_day)
        return {
            "schema_version": MARKET_CALENDAR_DECISION_SCHEMA_VERSION,
            "date": current.isoformat(),
            "market_status": "half_day" if is_half_day else "normal",
            "should_run": should_run,
            "run_mode": "half_day" if is_half_day and should_run else "regular" if should_run else "skipped_half_day",
            "skip_reason": "" if should_run else "half_day_disabled",
            "next_trading_date": current.isoformat(),
            "previous_trading_date": _day_string(previous_trading_date),
            "allow_half_day": bool(allow_half_day),
        }

    return {
        "schema_version": MARKET_CALENDAR_DECISION_SCHEMA_VERSION,
        "date": current.isoformat(),
        "market_status": "closed",
        "should_run": False,
        "run_mode": "closed",
        "skip_reason": reasons.get(current.isoformat(), "not_in_trading_calendar"),
        "next_trading_date": _day_string(next_future_trading_date),
        "previous_trading_date": _day_string(previous_trading_date),
        "allow_half_day": bool(allow_half_day),
    }
