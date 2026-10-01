"""Read fresh SOXL/SOXX opening quotes without any trading connection.

Futu names the US 09:30–16:00 continuous session ``AFTERNOON``, including
the opening minutes. ``MORNING`` is not the US equities regular-session state.
Authority: https://openapi.futunn.com/futu-api-doc/qa/quote.html (Q14).
The observation window and quote timestamps still determine timeliness.
"""
from __future__ import annotations

from datetime import datetime
import math
from zoneinfo import ZoneInfo


NY = ZoneInfo("America/New_York")
SYMBOLS = ("US.SOXL", "US.SOXX")
FUTU_MARKET_STATE_DOCS = "https://openapi.futunn.com/futu-api-doc/qa/quote.html"


def read_session_quotes(*, host: str, port: int, slot: str, now: datetime | None = None):
    """Return fresh quotes in the open or first-hour recovery observation slot.

    Futu's naive US snapshot timestamps are New York local time. Every quote
    must be within 120 seconds of this observation, allowing at most five
    seconds of clock skew. The market snapshot timestamp is not a fill time.
    """
    current = now or datetime.now(NY)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("needs_data: opening quote observation clock requires a timezone")
    current = current.astimezone(NY)
    if slot not in {"open", "recovery"}:
        raise ValueError("needs_data: unsupported quote observation slot")
    lower = 571 if slot == "open" else 631
    if current.weekday() >= 5 or not lower <= current.hour * 60 + current.minute < lower + 6:
        raise ValueError(f"needs_data: {slot} quote observation is outside its six-minute ET window")

    from futu import OpenQuoteContext, RET_OK

    context = OpenQuoteContext(host=host, port=port)
    try:
        status, state = context.get_global_state()
        if status != RET_OK:
            raise ValueError("needs_data: US market-state query failed")
        market_state = str(state.get("market_us", "")).upper()
        if market_state != "AFTERNOON":
            raise ValueError(f"needs_data: US regular session is not open (market_us={market_state})")
        status, frame = context.get_market_snapshot(list(SYMBOLS))
        if status != RET_OK:
            raise ValueError("needs_data: opening quote snapshot failed")
        prices, stamps = {}, []
        for row in frame.to_dict("records"):
            symbol = row.get("code")
            if symbol not in SYMBOLS:
                continue
            if symbol in prices:
                raise ValueError("needs_data: duplicate opening quote symbol")
            try:
                price = float(row["last_price"])
                stamp = datetime.fromisoformat(str(row["update_time"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("needs_data: invalid opening quote price or timestamp") from error
            if not math.isfinite(price) or price <= 0:
                raise ValueError("needs_data: invalid opening quote price")
            if stamp.tzinfo is None or stamp.utcoffset() is None:
                stamp = stamp.replace(tzinfo=NY)
            stamp = stamp.astimezone(NY)
            age = (current - stamp).total_seconds()
            if (stamp.date() != current.date()
                    or not 570 <= stamp.hour * 60 + stamp.minute < 960
                    or age < -5 or age > 120):
                raise ValueError("needs_data: stale, future, or off-session opening quote")
            prices[symbol] = price
            stamps.append(stamp)
        if set(prices) != set(SYMBOLS):
            raise ValueError("needs_data: incomplete SOXL/SOXX opening quote snapshot")
        return {symbol.removeprefix("US."): price for symbol, price in prices.items()}, min(stamps).isoformat()
    finally:
        context.close()


def read_open_quotes(*, host: str, port: int, now: datetime | None = None):
    """Compatibility signature for the sealed one-step opening scheduler."""
    return read_session_quotes(host=host, port=port, slot="open", now=now)
