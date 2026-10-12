"""Completed-session minute information for a next-open daily allocator."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _native_groups(frame: pd.DataFrame) -> dict:
    if frame.attrs.get("qlib_data_api") != ["qlib.data.D.features"]:
        raise ValueError("blocked: minute allocation inputs require native Qlib observations")
    values = frame.reset_index().rename(columns={c: c.removeprefix("$") for c in frame.columns})
    values["datetime"] = pd.to_datetime(values.datetime)
    values["session_date"] = values.datetime.dt.normalize()
    return dict(tuple(values.sort_values("datetime").groupby(["instrument", "session_date"], sort=False)))


def _qualified(part: pd.DataFrame | None, expected: pd.DatetimeIndex) -> bool:
    if part is None or part.datetime.duplicated().any() or not pd.DatetimeIndex(part.datetime).equals(expected):
        return False
    price = part[["open", "high", "low", "close", "volume"]].to_numpy(float)
    return bool(np.isfinite(price).all() and (price[:, :4] > 0).all() and (price[:, 4] >= 0).all()
                and part.price_valid.eq(1).all()
                and (price[:, 1] >= price[:, [0, 2, 3]].max(axis=1) - 1e-7).all()
                and (price[:, 2] <= price[:, [0, 1, 3]].min(axis=1) + 1e-7).all())


def _five_minute(part: pd.DataFrame) -> pd.DataFrame:
    bars = part.set_index("datetime").resample("5min", closed="right", label="right").agg(
        open=("open", "first"), close=("close", "last"), volume=("volume", "sum"))
    return bars.loc[bars.open.notna()].copy()


def _returns(part: pd.DataFrame) -> np.ndarray:
    previous = np.r_[float(part.open.iloc[0]), part.close.to_numpy(float)[:-1]]
    return np.log(part.close.to_numpy(float) / previous)


def _downside(returns: np.ndarray) -> float:
    variance = float(np.square(returns).sum())
    return float(np.square(returns[returns < 0]).sum() / variance) if variance else 0.0


def completed_session_features(core: pd.DataFrame, context: pd.DataFrame,
                               schedule: pd.DataFrame, *, core_names: list[str],
                               peer_names: list[str], peer_symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use raw NONE within-session ratios only; unknown channels stay NaN.

    Five-minute returns start with the observed regular-session open, so no
    overnight move or future corporate-action coefficient enters the path.
    Half days use their actual exchange close and relative closing windows.
    """
    core_groups, peers = _native_groups(core), _native_groups(context)
    rows, coverage = [], []
    for date, session in schedule.iterrows():
        day = pd.Timestamp(date).tz_localize(None).normalize()
        opening = pd.Timestamp(session["open"]).tz_convert("America/New_York").tz_localize(None)
        closing = pd.Timestamp(session["close"]).tz_convert("America/New_York").tz_localize(None)
        minutes = pd.date_range(opening + pd.Timedelta(minutes=1), closing, freq="min")
        five = pd.date_range(opening + pd.Timedelta(minutes=5), closing, freq="5min")
        row = {"datetime": day, "minute_core_ready": 0., "minute_peer_ready": 0.,
               "minute_feature_available_at": pd.Timestamp(session["close"]) + pd.Timedelta(minutes=1)}
        row.update({name: np.nan for name in [*core_names, *peer_names]})
        parts = {s: core_groups.get((s, day)) for s in ("SOXL", "SOXX")}
        core_ready = all(_qualified(part, minutes) for part in parts.values())
        peer_parts = {s: peers.get((s, day)) for s in peer_symbols}
        peer_ready = core_ready and all(_qualified(part, five) for part in peer_parts.values())
        coverage.append({"date": day, "expected_1m": len(minutes), "expected_5m": len(five),
                         "core_ready": core_ready, "peer_price_ready": peer_ready,
                         "early_close": len(minutes) < 390})
        if core_ready:
            sx, sl = parts["SOXX"], parts["SOXL"]
            bxx, bxl = _five_minute(sx), _five_minute(sl)
            rx, rl = _returns(bxx), _returns(bxl)
            r1 = _returns(sx)
            rv5 = float(np.square(rx).sum())
            late = bxx.index > closing - pd.Timedelta(minutes=60)
            last30 = bxx.index > closing - pd.Timedelta(minutes=30)
            first_hour_end = opening + pd.Timedelta(minutes=60)
            closes = np.r_[float(sx.open.iloc[0]), sx.close.to_numpy(float)]
            volume = float(sx.volume.sum())
            row.update({
                "minute_soxx_log_rv5": .5 * np.log(max(rv5, 1e-16)),
                "minute_soxx_downside_share5": _downside(rx),
                "minute_soxx_late_rv_share": float(np.square(rx[late]).sum() / rv5) if rv5 else 0.,
                "minute_soxx_trend_efficiency": float(rx.sum() / max(np.abs(rx).sum(), 1e-16)),
                "minute_soxx_post_first_hour_return": float(np.log(bxx.close.iloc[-1] / bxx.loc[first_hour_end, "close"])),
                "minute_soxx_late_return30": float(rx[last30].sum()),
                "minute_soxx_intraday_drawdown": float((closes / np.maximum.accumulate(closes) - 1).min()),
                "minute_soxx_rebound_from_low": float(closes[-1] / closes.min() - 1),
                "minute_soxx_late_volume_share": float(sx.loc[sx.datetime > closing - pd.Timedelta(minutes=60), "volume"].sum() / volume) if volume else 0.,
                "minute_soxx_rv1_rv5_ratio_log": float(np.log(max(np.square(r1).sum(), 1e-16) / max(rv5, 1e-16))),
                "minute_soxl_tracking_residual_rv5": float(np.sqrt(np.square(rl - 3 * rx).sum())),
                "minute_soxl_soxx_sign_agreement": float((np.sign(rl) == np.sign(rx)).mean()),
            })
            row["minute_core_ready"] = float(np.isfinite([row[n] for n in core_names]).all())
            if not row["minute_core_ready"]:
                row.update({name: np.nan for name in core_names})
        if peer_ready and row["minute_core_ready"]:
            peer_returns = {s: _returns(peer_parts[s]) for s in peer_symbols}
            total = {s: float(r.sum()) for s, r in peer_returns.items()}
            rv = {s: float(np.square(r).sum()) for s, r in peer_returns.items()}
            correlation = (float(np.corrcoef(rx, peer_returns["SMH"])[0, 1])
                           if np.std(rx) > 1e-12 and np.std(peer_returns["SMH"]) > 1e-12 else np.nan)
            row.update({
                "minute_smh_qqq_log_rv_gap": .5 * np.log(max(rv["SMH"], 1e-16) / max(rv["QQQ"], 1e-16)),
                "minute_smh_qqq_intraday_return_gap": total["SMH"] - total["QQQ"],
                "minute_nvda_amd_mean_return": .5 * (total["NVDA"] + total["AMD"]),
                "minute_nvda_amd_late_return_gap": .5 * (peer_returns["NVDA"][late].sum() + peer_returns["AMD"][late].sum()) - rx[late].sum(),
                "minute_nvda_amd_return_dispersion": abs(total["NVDA"] - total["AMD"]),
                "minute_peer_sign_agreement": float(np.mean([np.sign(total[s]) == np.sign(rx.sum()) for s in peer_symbols])),
                "minute_soxx_smh_return_correlation": correlation,
                "minute_peer_downside_share": float(np.mean([_downside(r) for r in peer_returns.values()])),
            })
            row["minute_peer_ready"] = float(np.isfinite([row[n] for n in peer_names]).all())
            if not row["minute_peer_ready"]:
                row.update({name: np.nan for name in peer_names})
        rows.append(row)
    features = pd.DataFrame(rows).set_index("datetime").sort_index()
    return features, pd.DataFrame(coverage)


def label_maturity(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Bind H5 label maturity and next-open dates to the actual native calendar."""
    dates = pd.Series(pd.DatetimeIndex(calendar).normalize(), index=pd.DatetimeIndex(calendar).normalize())
    result = pd.DataFrame(index=frame.index)
    result["label_available_at"] = pd.to_datetime(dates.shift(-6).reindex(frame.index)).dt.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)
    result["return_available_at"] = pd.to_datetime(dates.shift(-2).reindex(frame.index)).dt.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=31)
    # Existing daily inputs include VIX's later close. Evaluate the completed
    # prior session before the next open, not at the cash market's 16:01 mark.
    result["decision_at"] = pd.to_datetime(frame.trade_date).dt.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=29)
    for column, offset in [("trade_date", -1), ("exit_date", -2)]:
        actual = dates.shift(offset).reindex(frame.index)
        if not frame[column].equals(actual.rename(column)):
            if not pd.to_datetime(frame[column]).reset_index(drop=True).equals(actual.reset_index(drop=True)):
                raise ValueError("blocked: primary dates disagree with native Qlib calendar")
    return result
