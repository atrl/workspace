"""Conditional, basis-consistent revenue surprises; unknown observations stay unknown."""
import numpy as np
import pandas as pd


def revenue_information(paired, calendar, *, preliminary_reports, active_sessions=20):
    calendar = pd.DatetimeIndex(calendar)
    if not calendar.is_unique or not calendar.is_monotonic_increasing or pd.to_datetime(paired.report_date).min() < calendar[0]:
        raise ValueError("full native calendar must include event history before model start")
    blocked = {(item["symbol"], item["report_date"]) for item in preliminary_reports}
    events = []
    for (symbol, date), group in paired.groupby(["symbol", "report_date"], sort=True):
        by_basis = group.set_index("basis")
        if not by_basis.index.is_unique or set(by_basis.index) != {"gaap", "non_gaap"}:
            raise ValueError("expected both issuer bases for every calendar event")
        gaap, non_gaap = by_basis.loc["gaap"], by_basis.loc["non_gaap"]
        estimate = gaap.consensus_revenue_million_usd
        values = [gaap.revenue_million_usd, non_gaap.revenue_million_usd, estimate]
        reason = "usable_under_declared_timing"
        if not np.isfinite(values).all() or min(values) <= 0:
            reason = "unknown_revenue_or_estimate"
        elif gaap.revenue_million_usd != non_gaap.revenue_million_usd:
            reason = "issuer_revenue_basis_differs"
        if (symbol, date) in blocked:
            reason = "earlier_preliminary_disclosure"
        position = int(calendar.searchsorted(pd.Timestamp(date), side="right"))
        # Prior-calendar observations are already dated, but no exact first-receipt claim.
        available = calendar[position] + pd.Timedelta(hours=9, minutes=31) if position < len(calendar) else pd.NaT
        known = reason == "usable_under_declared_timing"
        events.append({"symbol": symbol, "report_date": date, "available_at": available,
            "calendar_position": position, "known": known, "reason": reason,
            "surprise": float(gaap.revenue_million_usd / estimate - 1) if known else np.nan,
            "rounding_half_ratio": float(gaap.revenue_rounding_half_million / estimate) if known else np.nan})
    events = pd.DataFrame(events)
    state = pd.DataFrame(index=calendar)
    for symbol in sorted(events.symbol.unique()):
        source = events.loc[events.symbol.eq(symbol)].sort_values("available_at")
        prefix = symbol.lower()
        for field in ["surprise", "known", "age", "preliminary"]:
            state[prefix + "_" + field] = np.nan if field == "surprise" else 0.
        state[prefix + "_age"] = float(active_sessions)
        for number, day in enumerate(calendar):
            eligible = source.loc[source.available_at.le(day + pd.Timedelta(hours=9, minutes=46))]
            if eligible.empty:
                continue
            latest = eligible.iloc[-1]
            age = number - latest.calendar_position
            state.loc[day, prefix + "_age"] = min(age, active_sessions)
            if age >= active_sessions:
                continue
            state.loc[day, prefix + "_known"] = float(latest.known)
            state.loc[day, prefix + "_preliminary"] = float(latest.reason == "earlier_preliminary_disclosure")
            state.loc[day, prefix + "_surprise"] = latest.surprise
    state["datetime"] = state.index
    return state.reset_index(drop=True), events
