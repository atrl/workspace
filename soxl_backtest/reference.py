from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .backtest import SOXLBacktest


@dataclass(frozen=True)
class SoxlReferenceResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]


def _max_drawdown_pct(values: list[float]) -> float:
    peak = float("-inf")
    max_dd = 0.0
    for value in values:
        current = float(value)
        peak = max(peak, current)
        if peak > 0:
            max_dd = max(max_dd, (peak - current) / peak * 100.0)
    return max_dd


def replay_soxlbacktest_reference(
    *,
    start: str,
    end: str,
    initial_capital: float = 100_000.0,
    config_overrides: dict[str, Any] | None = None,
    quiet: bool = True,
) -> SoxlReferenceResult:
    params: dict[str, Any] = {
        "USE_LONGBRIDGE": False,
        "START_DATE": start,
        "END_DATE": end,
        "INITIAL_CAPITAL": initial_capital,
    }
    if config_overrides:
        params.update(config_overrides)

    backtest = SOXLBacktest(params)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            backtest.run()
    else:
        backtest.run()

    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(backtest.df["date"]).dt.date.astype(str),
            "price": backtest.price_history,
            "capital": backtest.capital_history,
            "position": backtest.position_history,
            "panic_score": backtest.panic_score_history,
            "hv_20": backtest.hv_20_history,
            "hv_30": backtest.hv_30_history,
            "vol_coef": backtest.vol_coef_history,
        }
    )
    trades = pd.DataFrame(backtest.trades)
    if trades.empty:
        trades = pd.DataFrame(columns=["date", "type", "price", "position", "score", "total_value"])

    first_price = float(backtest.df.iloc[0]["price"])
    last_price = float(backtest.df.iloc[-1]["price"])
    final_capital = float(backtest.capital_history[-1]) if backtest.capital_history else initial_capital
    metrics = {
        "requested_start": start,
        "start": pd.Timestamp(backtest.df.iloc[0]["date"]).date().isoformat(),
        "end": pd.Timestamp(backtest.df.iloc[-1]["date"]).date().isoformat(),
        "rows": int(len(backtest.df)),
        "initial_capital": float(initial_capital),
        "final_capital": final_capital,
        "soxl_buyhold_return_pct": (last_price / first_price - 1.0) * 100.0,
        "rulebase_return_pct": (final_capital / float(initial_capital) - 1.0) * 100.0,
        "rulebase_max_drawdown_pct": _max_drawdown_pct([float(value) for value in backtest.capital_history]),
        "average_position_pct": float(pd.Series(backtest.position_history).mean() * 100.0),
        "end_position_pct": float(backtest.position_history[-1] * 100.0) if backtest.position_history else 0.0,
        "trade_count": int(len(backtest.trades)),
    }
    return SoxlReferenceResult(daily=daily, trades=trades, metrics=metrics)
