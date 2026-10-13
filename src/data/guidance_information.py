"""Conditional information state from reviewed prospective-quarter comparisons."""
import numpy as np
import pandas as pd


def guidance_information(review, manifest, calendar, *, active_sessions=20):
    calendar = pd.DatetimeIndex(calendar)
    if not calendar.is_unique or not calendar.is_monotonic_increasing or pd.to_datetime(review.report_date).min() < calendar[0]:
        raise ValueError("calendar must include event history")
    decisions = calendar + pd.Timedelta(hours=9, minutes=46)
    sources = {(e["symbol"], e["report_date"]): e for e in manifest["entries"]}
    events = []
    for _, row in review.iterrows():
        day = pd.Timestamp(row.newsletter_date)
        source = sources.get((row.symbol, row.report_date))
        known = row.status in {"numeric_pair", "direction_only"}
        available = day + pd.Timedelta(hours=9, minutes=31)
        if source:
            clocks = [pd.Timestamp(source[name]) for name in ["created_at", "modified_at"]]
            if any(pd.isna(clock) or clock.tzinfo is None for clock in clocks):
                known = False
            else:
                latest = max(clocks).tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(minutes=1)
                available = max(available, latest)
        else:
            known = False
        position = int(decisions.searchsorted(available))
        numeric = known and bool(row.numeric_pair_present)
        events.append({"symbol": row.symbol, "report_date": row.report_date, "status": row.status,
            "available_at": available, "calendar_position": position, "direction_known": known,
            "numeric_known": numeric, "direction": row.direction if known else np.nan,
            "gap": row.reported_midpoint_gap if numeric else np.nan})
    events = pd.DataFrame(events)
    state = pd.DataFrame(index=calendar)
    for symbol in sorted(events.symbol.unique()):
        source = events.loc[events.symbol.eq(symbol)].sort_values(["available_at", "report_date"])
        prefix = symbol.lower()
        for name in ["direction", "gap", "direction_known", "numeric_known", "age"]:
            state[prefix + "_" + name] = np.nan if name in {"direction", "gap"} else float(active_sessions) if name == "age" else 0.
        for number, day in enumerate(calendar):
            available = source.loc[source.calendar_position.le(number)]
            if available.empty:
                continue
            latest = available.iloc[-1]; age = number - latest.calendar_position
            state.loc[day, prefix + "_age"] = min(age, active_sessions)
            if age >= active_sessions:
                continue
            for name in ["direction", "gap", "direction_known", "numeric_known"]:
                state.loc[day, prefix + "_" + name] = float(latest[name])
    state["datetime"] = state.index
    return state.reset_index(drop=True), events
