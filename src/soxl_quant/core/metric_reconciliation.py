from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pandas as pd

from .constants import ACCOUNT


def _trading_dates(index: pd.Index | pd.MultiIndex | Sequence[object]) -> pd.DatetimeIndex:
    raw_index = pd.Index(index)
    if isinstance(raw_index, pd.MultiIndex):
        level_name = "datetime" if "datetime" in raw_index.names else raw_index.names[0]
        values = raw_index.get_level_values(level_name)
    else:
        values = raw_index
    dates = pd.DatetimeIndex(pd.to_datetime(values)).dropna().unique().sort_values()
    if dates.empty:
        raise ValueError("empty trading calendar")
    return dates


def _date_label(value: pd.Timestamp) -> str:
    return str(pd.Timestamp(value).date())


def build_warm_start_replay_segment(
    index: pd.Index | pd.MultiIndex | Sequence[object],
    *,
    eval_seg: tuple[str, str],
    lookback: int,
) -> tuple[str, str]:
    """Return an official replay segment with prior trading rows for warmup.

    The returned segment starts ``lookback`` trading rows before the requested
    evaluation start, clamped to available data, and ends at the last available
    trading row within ``eval_seg``. Metrics should still be sliced back to
    ``eval_seg`` after the qlib replay completes.
    """

    dates = _trading_dates(index)
    eval_start = pd.Timestamp(eval_seg[0])
    eval_end = pd.Timestamp(eval_seg[1])
    eval_dates = dates[(dates >= eval_start) & (dates <= eval_end)]
    if eval_dates.empty:
        raise ValueError(f"empty evaluation window for segment {eval_seg}")
    eval_start_pos = int(dates.get_loc(eval_dates[0]))
    replay_start_pos = max(0, eval_start_pos - max(int(lookback), 0))
    return (_date_label(dates[replay_start_pos]), _date_label(eval_dates[-1]))


def _series_with_datetime_index(series: pd.Series) -> pd.Series:
    out = series.copy()
    if isinstance(out.index, pd.MultiIndex):
        level_name = "datetime" if "datetime" in out.index.names else out.index.names[0]
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index.get_level_values(level_name)))
    else:
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index))
    return out[~out.index.duplicated(keep="last")].sort_index()


def _frame_with_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.index, pd.MultiIndex):
        level_name = "datetime" if "datetime" in out.index.names else out.index.names[0]
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index.get_level_values(level_name)))
    else:
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index))
    return out[~out.index.duplicated(keep="last")].sort_index()


def slice_warm_start_daily_path(
    *,
    portfolio_df: pd.DataFrame,
    price_series: pd.Series,
    eval_seg: tuple[str, str],
    position_history: Sequence[float] | None = None,
    trade_signals: Sequence[tuple] | None = None,
    account_base: float = ACCOUNT,
) -> pd.DataFrame:
    """Slice a warm-start official replay to eval dates and rebase NAVs.

    qlib should be allowed to replay a warmup segment so the policy has enough
    history to issue orders. This helper makes the metric window explicit by
    cutting back to ``eval_seg`` and rebasing both strategy and buy-and-hold NAV
    from the first evaluation date.
    """

    portfolio = _frame_with_datetime_index(portfolio_df)
    if "account" not in portfolio.columns:
        raise ValueError("portfolio_df missing required 'account' column")
    account = portfolio["account"].astype(float)
    eval_account = account.loc[pd.Timestamp(eval_seg[0]) : pd.Timestamp(eval_seg[1])].dropna()
    if eval_account.empty:
        raise ValueError(f"empty evaluation account window for segment {eval_seg}")
    start_account = float(eval_account.iloc[0])
    if not math.isfinite(start_account) or start_account == 0.0:
        raise ValueError("invalid evaluation start account")

    prices = _series_with_datetime_index(price_series).astype(float)
    aligned_prices = prices.reindex(account.index).ffill().bfill()
    eval_prices = aligned_prices.reindex(eval_account.index).ffill().bfill()
    if eval_prices.isna().any():
        raise ValueError(f"price series does not cover evaluation segment {eval_seg}")
    start_price = float(eval_prices.iloc[0])
    if not math.isfinite(start_price) or start_price == 0.0:
        raise ValueError("invalid evaluation start price")

    strategy_nav = float(account_base) * (eval_account / start_account)
    buyhold_nav = float(account_base) * (eval_prices / start_price)

    if position_history:
        pos_values = [float(value) for value in position_history]
        pos_index = account.index[: len(pos_values)]
        position = pd.Series(pos_values, index=pos_index).reindex(eval_account.index).ffill().fillna(0.0)
    else:
        position = pd.Series(0.0, index=eval_account.index)

    action_map: dict[str, str] = {}
    for signal in trade_signals or []:
        if len(signal) < 4:
            continue
        dt, _price, _pos, signal_type = signal[:4]
        action_map[_date_label(pd.Timestamp(dt))] = str(signal_type).upper()

    daily = pd.DataFrame(
        {
            "datetime": [_date_label(dt) for dt in eval_account.index],
            "strategy_nav": strategy_nav.astype(float).values,
            "price": eval_prices.astype(float).values,
            "buyhold_nav": buyhold_nav.astype(float).values,
            "position": position.astype(float).values,
        }
    )
    daily["excess"] = daily["strategy_nav"] - daily["buyhold_nav"]
    daily["excess_pct"] = daily["excess"] / daily["buyhold_nav"] * 100.0
    daily["action"] = [action_map.get(dt, "HOLD") for dt in daily["datetime"]]
    return daily


def _window_prices(price_series: pd.Series, seg: tuple[str, str]) -> pd.Series:
    prices = price_series.sort_index().astype(float).loc[seg[0] : seg[1]].dropna()
    if prices.empty:
        raise ValueError(f"empty price window for segment {seg}")
    return prices


def _return_pct(prices: pd.Series) -> float:
    if len(prices) < 2:
        return 0.0
    start = float(prices.iloc[0])
    end = float(prices.iloc[-1])
    if not math.isfinite(start) or start == 0.0:
        raise ValueError("invalid benchmark start price")
    return float((end / start - 1.0) * 100.0)


def compute_relative_metric_variants(
    *,
    strategy_return: float,
    price_series: pd.Series,
    requested_seg: tuple[str, str],
    effective_seg: tuple[str, str] | None = None,
    lookback: int = 0,
    target_relative_return: float = 30.0,
    material_threshold_pp: float = 5.0,
) -> dict[str, Any]:
    """Compare calendar and tradable-window relative-return metrics.

    Isolated qlib fold backtests often pass a slice beginning at the evaluation
    start date. The strategy then cannot issue orders for the first ``lookback``
    rows, while a calendar buy-and-hold benchmark can still include that warmup
    period. This helper keeps both benchmark views explicit so notebooks and
    promotion gates can label warmup-driven relative-return flips instead of
    silently mixing metric definitions.
    """

    effective = effective_seg or requested_seg
    calendar_prices = _window_prices(price_series, effective)
    requested_prices = _window_prices(price_series, requested_seg)
    end_date = calendar_prices.index[-1]
    requested_until_effective_end = requested_prices.loc[:end_date]
    if requested_until_effective_end.empty:
        requested_until_effective_end = calendar_prices

    first_tradable_idx = min(max(int(lookback), 0), len(requested_until_effective_end) - 1)
    first_tradable_date = requested_until_effective_end.index[first_tradable_idx]
    tradable_prices = requested_until_effective_end.loc[first_tradable_date:end_date]
    if len(tradable_prices) < 2:
        tradable_prices = calendar_prices
        first_tradable_date = tradable_prices.index[0]
        first_tradable_idx = 0

    calendar_bnh = _return_pct(calendar_prices)
    tradable_bnh = _return_pct(tradable_prices)
    calendar_relative = float(strategy_return) - calendar_bnh
    tradable_relative = float(strategy_return) - tradable_bnh
    calendar_pass = calendar_relative >= float(target_relative_return)
    tradable_pass = tradable_relative >= float(target_relative_return)
    gap = tradable_relative - calendar_relative
    gate_flip = bool(calendar_pass != tradable_pass)

    if gate_flip:
        reason = "warmup_benchmark_alignment_gate_flip"
    elif abs(gap) >= float(material_threshold_pp):
        reason = "material_warmup_benchmark_alignment_gap"
    else:
        reason = "metric_variants_aligned"

    return {
        "strategy_return": float(strategy_return),
        "calendar_buy_and_hold_return": float(calendar_bnh),
        "tradable_buy_and_hold_return": float(tradable_bnh),
        "calendar_relative_return": float(calendar_relative),
        "tradable_relative_return": float(tradable_relative),
        "relative_metric_gap": float(gap),
        "target_relative_return": float(target_relative_return),
        "calendar_gate": bool(calendar_pass),
        "tradable_gate": bool(tradable_pass),
        "gate_flip": gate_flip,
        "mismatch_reason": reason,
        "requested_start": str(pd.Timestamp(requested_prices.index[0]).date()),
        "requested_end": str(pd.Timestamp(requested_prices.index[-1]).date()),
        "effective_start": str(pd.Timestamp(calendar_prices.index[0]).date()),
        "effective_end": str(pd.Timestamp(calendar_prices.index[-1]).date()),
        "first_tradable_date": str(pd.Timestamp(first_tradable_date).date()),
        "warmup_days": int(first_tradable_idx),
    }


__all__ = [
    "build_warm_start_replay_segment",
    "compute_relative_metric_variants",
    "slice_warm_start_daily_path",
]
