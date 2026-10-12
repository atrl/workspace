"""Minute coverage against an explicit exchange schedule, before Qlib ingestion.

An observed-bar count cannot certify missing minutes or an absent session. The
caller supplies the exchange schedule and timestamp convention; this module
does not infer either from the input's first/last bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def audit_minute_coverage(
    frame: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    symbols: list[str],
    timestamp_convention: str,
    observed_at: pd.Timestamp,
) -> pd.DataFrame:
    """Audit closed sessions; in-progress sessions cannot become training-ready.

    ``schedule`` has timezone-aware ``open``/``close`` columns and session-date
    indices. ``right_edge`` means the first minute ends at open + one minute.
    Zero-volume bars are retained. Bad OHLC, duplicate keys and missing minutes
    invalidate a closed symbol-session. Turnover/price basis is a separate flag.
    """
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    if required - set(frame):
        raise ValueError(f"minute audit missing columns: {sorted(required - set(frame))}")
    if timestamp_convention != "right_edge":
        raise ValueError("blocked: explicit verified right_edge convention required")
    observed_at = pd.Timestamp(observed_at)
    if observed_at.tzinfo is None:
        raise ValueError("blocked: observed_at must be timezone-aware")
    if not {"open", "close"}.issubset(schedule) or schedule.index.has_duplicates:
        raise ValueError("blocked: unique explicit exchange schedule required")
    data = frame.copy()
    stamps = pd.to_datetime(data.timestamp)
    if not isinstance(stamps.dtype, pd.DatetimeTZDtype):
        raise ValueError("blocked: minute timestamps must be timezone-aware")
    data["timestamp"] = stamps.dt.tz_convert("America/New_York")
    data["symbol"] = data.symbol.astype(str).str.upper()
    data["session_date"] = data.timestamp.dt.strftime("%Y-%m-%d")
    if "interval" in data and not data.interval.eq("1m").all():
        raise ValueError("blocked: minute audit accepts only 1m bars")
    prices = data[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    volume = pd.to_numeric(data.volume, errors="coerce")
    data["valid"] = (
        np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
        & np.isfinite(volume) & volume.ge(0)
        & prices.high.ge(prices.max(axis=1) - 1e-8)
        & prices.low.le(prices.min(axis=1) + 1e-8)
    )
    data["vwap_checked"] = False
    data["vwap_outside"] = False
    if "turnover" in data:
        vwap = pd.to_numeric(data.turnover, errors="coerce") / volume.replace(0, np.nan)
        data["vwap_checked"] = np.isfinite(vwap) & volume.gt(0)
        # 0.1% tolerance is a basis diagnostic, not an automatic price repair.
        data["vwap_outside"] = data.vwap_checked & (
            vwap.lt(prices.low * 0.999) | vwap.gt(prices.high * 1.001)
        )
    groups = dict(tuple(data.groupby(["symbol", "session_date"], sort=False)))
    empty = data.iloc[:0]
    rows = []
    for session, hours in schedule.sort_index().iterrows():
        session_date = pd.Timestamp(session).strftime("%Y-%m-%d")
        opening, closing = pd.Timestamp(hours.open), pd.Timestamp(hours.close)
        if opening.tzinfo is None or closing.tzinfo is None or closing <= opening:
            raise ValueError("blocked: invalid exchange session hours")
        opening, closing = opening.tz_convert("America/New_York"), closing.tz_convert("America/New_York")
        grid = pd.date_range(opening + pd.Timedelta(minutes=1), closing, freq="min")
        due = grid[grid <= observed_at.floor("min")]
        closed = observed_at > closing
        for symbol in sorted(set(s.upper() for s in symbols)):
            bars = groups.get((symbol, session_date), empty)
            bars = bars.loc[bars.timestamp.isin(due)]
            unique = pd.DatetimeIndex(bars.timestamp.drop_duplicates())
            missing = due.difference(unique)
            duplicates = int(bars.timestamp.duplicated().sum())
            invalid = int((~bars.valid).sum())
            status = "in_progress" if not closed else (
                "complete" if not len(missing) and not duplicates and not invalid else "needs_data"
            )
            rows.append({
                "symbol": symbol, "date": session_date, "expected_minutes": len(grid),
                "due_minutes": len(due), "observed_minutes": len(unique),
                "missing_minutes": len(missing), "duplicate_rows": duplicates,
                "invalid_rows": invalid, "zero_volume_rows": int(bars.volume.eq(0).sum()),
                "vwap_checked_rows": int(bars.vwap_checked.sum()),
                "vwap_outside_rows": int(bars.vwap_outside.sum()),
                "first_missing_at": str(missing[0]) if len(missing) else None,
                "last_missing_at": str(missing[-1]) if len(missing) else None,
                "status": status, "training_eligible": status == "complete",
                "repair_required": closed and status != "complete",
            })
    return pd.DataFrame(rows)


def plan_minute_repairs(coverage: pd.DataFrame) -> pd.DataFrame:
    """Return explicit per-symbol/day history requests, including older holes.

    Fetching succeeds independently of admission: verify adjustment vintage,
    overlapping prices/volumes and closed-bar status before merging a repair.
    """
    repair = coverage.loc[coverage.repair_required].copy()
    repair["start"] = repair.date
    repair["end"] = repair.date
    repair["admission"] = "pending_price_basis_and_overlap_validation"
    return repair[["symbol", "start", "end", "missing_minutes", "invalid_rows", "admission"]].reset_index(drop=True)
