from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
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


def _read_qlib_calendar(path: str | Path) -> list[str]:
    calendar_path = Path(path)
    return [line.strip()[:10] for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metadata_date_values(payload: Mapping[str, Any], key: str) -> list[str]:
    values = payload.get(key, ())
    if isinstance(values, Mapping):
        return [str(value)[:10] for value in values.keys()]
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        return []
    dates: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            value = item.get("date")
        else:
            value = item
        if value:
            dates.append(str(value)[:10])
    return dates


def _metadata_holiday_reasons(payload: Mapping[str, Any]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    values = payload.get("holiday_reasons", {})
    if isinstance(values, Mapping):
        reasons.update({str(key)[:10]: str(value) for key, value in values.items()})
    holidays = payload.get("holidays", ())
    if isinstance(holidays, Mapping):
        reasons.update({str(key)[:10]: str(value) for key, value in holidays.items()})
        return reasons
    if isinstance(holidays, Iterable) and not isinstance(holidays, (str, bytes)):
        for item in holidays:
            if not isinstance(item, Mapping) or not item.get("date"):
                continue
            reason = item.get("reason") or item.get("name") or item.get("label") or "market_holiday"
            reasons[str(item["date"])[:10]] = str(reason)
    return reasons


def _read_market_calendar_metadata(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("market calendar metadata must be a JSON object")
    return payload


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


def load_market_calendar_decision(
    as_of: object,
    *,
    calendar_path: str | Path,
    metadata_path: str | Path | None = None,
    allow_half_day: bool = True,
) -> dict[str, Any]:
    """Load qlib calendar files plus reviewable sidecar metadata into a schedule decision."""

    metadata = _read_market_calendar_metadata(metadata_path)
    decision = classify_market_calendar_day(
        as_of,
        trading_calendar=_read_qlib_calendar(calendar_path),
        half_days=_metadata_date_values(metadata, "half_days"),
        holiday_reasons=_metadata_holiday_reasons(metadata),
        allow_half_day=allow_half_day,
    )
    decision.update(
        {
            "calendar_source": str(calendar_path),
            "metadata_source": str(metadata_path) if metadata_path is not None else "",
            "exchange": str(metadata.get("exchange", "")),
            "timezone": str(metadata.get("timezone", "")),
            "metadata_schema_version": str(metadata.get("schema_version", "")),
        }
    )
    return decision
