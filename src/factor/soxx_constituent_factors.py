from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, load_qlib_feature_frame


SOXX_CONSTITUENT_FACTOR_SOURCE = "qlib.data.D.features"
SOXX_CONSTITUENT_LIMITATIONS = [
    "Current SOXX constituents backfilled through history are survivorship-biased without point-in-time holdings.",
    "All constituent return aggregates are lagged before use as open-to-open model features.",
]


def build_soxx_constituent_factor_frame(
    config: QlibProviderConfig,
    *,
    instruments: Iterable[str],
    start: str,
    end: str,
    lookbacks: tuple[int, ...] = (5, 20),
    lag_days: int = 1,
    top_n: int = 5,
) -> pd.DataFrame:
    """Build lagged SOXX constituent breadth and dispersion factors from a qlib provider."""

    instrument_list = [str(instrument).upper() for instrument in instruments]
    if not instrument_list:
        raise ValueError("SOXX constituent factor frame requires at least one instrument")
    if lag_days < 1:
        raise ValueError("SOXX constituent factors must use lag_days >= 1 for open-to-open execution")
    if any(int(lookback) < 1 for lookback in lookbacks):
        raise ValueError("SOXX constituent factor lookbacks must be positive")
    if top_n < 1:
        raise ValueError("SOXX constituent top_n must be positive")

    qlib_frame = load_qlib_feature_frame(
        config,
        instruments=instrument_list,
        fields=["$adj_close", "$close", "$volume", "$weight_pct"],
        start=start,
        end=end,
    )
    panel = qlib_frame.reset_index().copy()
    panel["instrument"] = panel["instrument"].astype(str).str.upper()
    panel["datetime"] = pd.to_datetime(panel["datetime"]).dt.normalize()
    price_col = "$adj_close" if "$adj_close" in panel.columns else "$close"
    panel["price"] = pd.to_numeric(panel[price_col], errors="coerce")
    panel["weight_pct"] = pd.to_numeric(panel.get("$weight_pct"), errors="coerce")
    panel = panel.sort_values(["instrument", "datetime"]).reset_index(drop=True)

    calendar = pd.Index(sorted(panel["datetime"].dropna().unique()), name="datetime")
    feature_frame = pd.DataFrame({"datetime": calendar})
    for lookback in tuple(int(item) for item in lookbacks):
        ret_col = f"ret_{lookback}"
        panel[ret_col] = panel.groupby("instrument", sort=False)["price"].pct_change(
            lookback,
            fill_method=None,
        )
        daily = panel.loc[panel[ret_col].notna() & panel["weight_pct"].notna()].copy()
        prefix = f"soxx_const"
        columns = {
            "breadth": f"{prefix}_breadth_ret{lookback}_pos_lag{lag_days}",
            "downside": f"{prefix}_downside_ret{lookback}_lt_m5_lag{lag_days}",
            "weighted": f"{prefix}_weighted_ret{lookback}_lag{lag_days}",
            "equal": f"{prefix}_equal_ret{lookback}_lag{lag_days}",
            "dispersion": f"{prefix}_dispersion_ret{lookback}_lag{lag_days}",
            "top_minus_rest": f"{prefix}_top{top_n}_minus_rest_ret{lookback}_lag{lag_days}",
        }
        aggregated = daily.groupby("datetime", sort=True).apply(
            lambda group: pd.Series(
                _aggregate_constituent_returns(group, ret_col=ret_col, top_n=top_n),
                index=[
                    columns["breadth"],
                    columns["downside"],
                    columns["weighted"],
                    columns["equal"],
                    columns["dispersion"],
                    columns["top_minus_rest"],
                ],
            ),
            include_groups=False,
        )
        aggregated = aggregated.reindex(calendar).shift(lag_days)
        feature_frame = feature_frame.merge(aggregated.reset_index(), on="datetime", how="left")

    count = (
        panel.loc[panel["price"].notna()]
        .groupby("datetime", sort=True)["instrument"]
        .nunique()
        .reindex(calendar)
        .astype("float64")
        .shift(lag_days)
        .rename(f"soxx_const_available_count_lag{lag_days}")
    )
    feature_frame = feature_frame.merge(count.reset_index(), on="datetime", how="left")
    feature_cols = [column for column in feature_frame.columns if column != "datetime"]
    feature_frame.attrs["feature_cols"] = feature_cols
    feature_frame.attrs["source"] = SOXX_CONSTITUENT_FACTOR_SOURCE
    feature_frame.attrs["provider_uri"] = config.resolved_provider_uri.as_posix()
    feature_frame.attrs["instruments"] = instrument_list
    feature_frame.attrs["lookbacks"] = list(lookbacks)
    feature_frame.attrs["lag_days"] = int(lag_days)
    feature_frame.attrs["top_n"] = int(top_n)
    feature_frame.attrs["limitations"] = SOXX_CONSTITUENT_LIMITATIONS
    return feature_frame


def _aggregate_constituent_returns(group: pd.DataFrame, *, ret_col: str, top_n: int) -> list[float]:
    returns = pd.to_numeric(group[ret_col], errors="coerce")
    weights = pd.to_numeric(group["weight_pct"], errors="coerce").clip(lower=0.0)
    valid = returns.notna() & weights.notna() & weights.gt(0.0)
    returns = returns.loc[valid].astype("float64")
    weights = weights.loc[valid].astype("float64")
    if returns.empty or weights.sum() <= 0.0:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    normalized_weights = weights / weights.sum()
    weighted_return = float((returns * normalized_weights).sum())
    equal_return = float(returns.mean())
    breadth = float(returns.gt(0.0).mean())
    downside = float(returns.lt(-0.05).mean())
    dispersion = float(np.sqrt(((returns - weighted_return) ** 2 * normalized_weights).sum()))

    ranked = pd.DataFrame({"return": returns, "weight": weights}).sort_values("weight", ascending=False)
    top = ranked.head(top_n)
    rest = ranked.iloc[top_n:]
    top_return = _weighted_average(top)
    rest_return = _weighted_average(rest)
    top_minus_rest = np.nan if np.isnan(top_return) or np.isnan(rest_return) else float(top_return - rest_return)
    return [breadth, downside, weighted_return, equal_return, dispersion, top_minus_rest]


def _weighted_average(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    weights = pd.to_numeric(frame["weight"], errors="coerce").clip(lower=0.0)
    returns = pd.to_numeric(frame["return"], errors="coerce")
    valid = weights.notna() & returns.notna() & weights.gt(0.0)
    if not valid.any():
        return np.nan
    weights = weights.loc[valid]
    returns = returns.loc[valid]
    return float((returns * (weights / weights.sum())).sum())
