"""First-15-minute paired ETF volume timing from native Qlib bars."""
from __future__ import annotations

import numpy as np
import pandas as pd


def volume_timing_residual(bars: pd.DataFrame, trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return one 09:46-known coordinate; incomplete dates stay explicitly unknown.

    Each ETF's volume-weighted mean minute log return minus its unweighted mean
    removes the aggregate signed first-15-minute move. The paired difference
    is a cross-ETF pressure proxy, not a tracking-error estimate.
    """
    if not isinstance(bars.index, pd.MultiIndex):
        raise ValueError("blocked: Qlib minute bars need instrument/datetime index")
    data = bars.reset_index().rename(columns={"$open": "open", "$close": "close", "$volume": "volume"})
    data["datetime"] = pd.to_datetime(data.datetime)
    data["trade_date"] = data.datetime.dt.normalize()
    clock = data.datetime.dt.strftime("%H:%M")
    data = data.loc[clock.between("09:31", "09:45")].copy()
    output = []
    for day in pd.DatetimeIndex(trade_dates):
        pieces = {}
        for symbol in ("SOXL", "SOXX"):
            part = data.loc[data.trade_date.eq(day) & data.instrument.eq(symbol)].sort_values("datetime")
            expected = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=15, freq="1min")
            valid = (len(part) == 15 and pd.DatetimeIndex(part.datetime).equals(expected)
                and np.isfinite(part[["open", "close", "volume"]].to_numpy(float)).all()
                and part[["open", "close"]].gt(0).all().all()
                and part.volume.ge(0).all() and part.volume.sum() > 0)
            if not valid:
                pieces = {}
                break
            minute = np.log(part.close.to_numpy(float) / part.open.to_numpy(float))
            dollar = part.open.to_numpy(float) * part.volume.to_numpy(float)
            pieces[symbol] = (float(np.average(minute, weights=dollar) - minute.mean()),
                              float(minute.sum()))
        known = len(pieces) == 2
        output.append({"trade_date": day, "volume_timing_residual":
            pieces["SOXL"][0] - 3 * pieces["SOXX"][0] if known else np.nan,
            "soxl_early_log_return": pieces["SOXL"][1] if known else np.nan,
            "soxx_early_log_return": pieces["SOXX"][1] if known else np.nan,
            "source_known": bool(known),
            "first_available_at": day + pd.Timedelta(hours=9, minutes=46) if known else pd.NaT})
    return pd.DataFrame(output)
