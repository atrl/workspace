from __future__ import annotations

import math
from typing import Any

import pandas as pd


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


__all__ = ["compute_relative_metric_variants"]
