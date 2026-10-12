"""Descriptive volatility summaries of native Qlib minute feature outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def minute_volatility_reports(
    features: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    segments: list[tuple[str, int, int]],
) -> dict[str, pd.DataFrame]:
    """Analyze complete paired regular sessions; never fit or select a strategy.

    Segment endpoints are minutes since midnight ET, (left, right]. Realized
    variance is sum(log-return squared), including first close / session open.
    A five-minute estimator is computed from the same native minute prices.
    """
    if features.attrs.get("qlib_data_api") != ["qlib.data.D.features"]:
        raise ValueError("blocked: native Qlib minute feature output required")
    if not isinstance(features.index, pd.MultiIndex) or not {"instrument", "datetime"}.issubset(features.index.names):
        raise ValueError("blocked: Qlib instrument/datetime index required")
    needed = ["$open", "$high", "$low", "$close", "$volume"]
    if not set(needed).issubset(features):
        raise ValueError("blocked: Qlib OHLCV features missing")
    if not segments or any(b <= a for _, a, b in segments):
        raise ValueError("blocked: nonempty positive-duration segments required")
    ordered = sorted(segments, key=lambda x: x[1])
    if any(left[2] != right[1] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("blocked: segments must form one contiguous session partition")
    eligibility = coverage.groupby("date").training_eligible.all()
    full = coverage.groupby("date").expected_minutes.apply(lambda n: n.eq(ordered[-1][2] - ordered[0][1]).all())
    valid_dates = eligibility[eligibility & full].index
    bars = features.reset_index().rename(columns={c: c.lstrip("$") for c in needed})
    bars["date"] = pd.to_datetime(bars.datetime).dt.strftime("%Y-%m-%d")
    bars = bars.loc[bars.date.isin(valid_dates)].sort_values(["instrument", "datetime"])
    values = bars[[name.lstrip("$") for name in needed]].to_numpy()
    if bars.empty or not np.isfinite(values).all():
        raise ValueError("blocked: missing native minute data on eligible sessions")
    if bars[["open", "high", "low", "close"]].le(0).any().any() or bars.volume.lt(0).any():
        raise ValueError("blocked: invalid native price on eligible sessions")
    actual = bars.groupby(["instrument", "date"]).size()
    expected = coverage.set_index(["symbol", "date"]).expected_minutes.reindex(actual.index)
    if expected.isna().any() or not actual.eq(expected).all():
        raise ValueError("blocked: native Qlib and source minute coverage differ")
    day_group = bars.groupby(["instrument", "date"], sort=False)
    previous = day_group.close.shift(1).fillna(day_group.open.transform("first"))
    bars["log_return"] = np.log(bars.close / previous)
    bars["rv"] = bars.log_return.pow(2)
    bars["downside_rv"] = bars.rv.where(bars.log_return.lt(0), 0.0)
    bars["clock"] = pd.to_datetime(bars.datetime).dt.hour * 60 + pd.to_datetime(bars.datetime).dt.minute
    daily = day_group.agg(
        rv=("rv", "sum"), downside_rv=("downside_rv", "sum"),
        volume=("volume", "sum"), opening=("open", "first"), closing=("close", "last"),
        high=("high", "max"), low=("low", "min"), minutes=("close", "size"),
    )
    daily["realized_vol_pct"] = 100 * np.sqrt(daily.rv)
    daily["range_pct"] = 100 * (daily.high / daily.low - 1)
    daily["open_close_pct"] = 100 * (daily.closing / daily.opening - 1)
    daily["downside_share"] = daily.downside_rv / daily.rv.replace(0, np.nan)
    segment_rows = []
    for name, start, end in ordered:
        part = bars.loc[bars.clock.gt(start) & bars.clock.le(end)]
        grouped = part.groupby(["instrument", "date"]).agg(
            rv=("rv", "sum"), downside_rv=("downside_rv", "sum"),
            volume=("volume", "sum"), opening=("open", "first"), closing=("close", "last"),
        )
        grouped["segment"] = name
        grouped["minutes"] = end - start
        grouped["rv_share"] = grouped.rv / daily.rv
        grouped["volume_share"] = grouped.volume / daily.volume.replace(0, np.nan)
        grouped["vol_per_minute_bps"] = 10000 * np.sqrt(grouped.rv / (end - start))
        grouped["return_pct"] = 100 * (grouped.closing / grouped.opening - 1)
        segment_rows.append(grouped.reset_index())
    per_day_segment = pd.concat(segment_rows, ignore_index=True)
    profile = per_day_segment.groupby(["instrument", "segment"], sort=False).agg(
        sessions=("date", "size"), minutes=("minutes", "first"),
        mean_variance_share=("rv_share", "mean"), median_variance_share=("rv_share", "median"),
        mean_volume_share=("volume_share", "mean"),
        median_vol_per_minute_bps=("vol_per_minute_bps", "median"),
    ).reset_index()
    coarse = bars.loc[bars.clock.mod(5).eq(0)].copy()
    coarse_group = coarse.groupby(["instrument", "date"])
    coarse_previous = coarse_group.close.shift(1).fillna(coarse_group.open.transform("first"))
    # The first five-minute return starts at the actual session open, not at
    # the fifth minute's open.
    first = coarse_group.cumcount().eq(0)
    day_open = daily.opening.reindex(pd.MultiIndex.from_frame(coarse[["instrument", "date"]])).to_numpy()
    coarse_previous.loc[first] = day_open[first.to_numpy()]
    coarse["rv5"] = np.log(coarse.close / coarse_previous).pow(2)
    daily["rv5"] = coarse.groupby(["instrument", "date"]).rv5.sum()
    daily["rv1_to_rv5_ratio"] = daily.rv / daily.rv5.replace(0, np.nan)
    pivot = bars.pivot(index="datetime", columns="instrument", values="log_return")
    paired = pd.DataFrame(index=pivot.index)
    if {"SOXL", "SOXX"}.issubset(pivot):
        paired["tracking_residual"] = pivot.SOXL - 3.0 * pivot.SOXX
        paired["residual_squared"] = paired.tracking_residual.pow(2)
        paired["date"] = pd.to_datetime(paired.index).strftime("%Y-%m-%d")
        residual = paired.groupby("date").residual_squared.sum().pow(0.5) * 100
    else:
        residual = pd.Series(dtype=float, name="residual_vol_pct")
    annual = daily.reset_index()
    annual["year"] = pd.to_datetime(annual.date).dt.year
    annual = annual.groupby(["instrument", "year"]).agg(
        sessions=("date", "size"), median_realized_vol_pct=("realized_vol_pct", "median"),
        p90_realized_vol_pct=("realized_vol_pct", lambda x: x.quantile(0.9)),
        median_range_pct=("range_pct", "median"), median_rv1_to_rv5_ratio=("rv1_to_rv5_ratio", "median"),
    ).reset_index()
    residual.name = "residual_vol_pct"
    return {"daily": daily.reset_index(), "segment_daily": per_day_segment, "profile": profile,
            "annual": annual, "residual": residual.reset_index()}
