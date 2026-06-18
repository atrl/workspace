from __future__ import annotations

import numpy as np
import pandas as pd


def _prediction_to_position(prediction: pd.Series) -> pd.Series:
    centered = prediction.astype(float)
    scale = centered.abs().median()
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = centered.abs().mean()
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    raw = 0.5 + centered / (4.0 * scale)
    return raw.clip(lower=0.0, upper=1.0)


def backtest_prediction_frame(predictions: pd.DataFrame, *, price_frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "model", "prediction", "target_return"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"prediction frame missing columns: {missing}")
    prices = price_frame.loc[:, ["date", "close", "vix"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    rows: list[pd.DataFrame] = []
    for model, group in predictions.copy().groupby("model", sort=True):
        data = group.copy()
        data["date"] = pd.to_datetime(data["date"])
        data = data.merge(prices, on="date", how="left").sort_values("date")
        position = _prediction_to_position(data["prediction"])
        turnover = position.diff().abs().fillna(0.0)
        realized = data["target_return"].astype(float).fillna(0.0)
        strategy_return = (position * realized).shift(1).fillna(0.0)
        benchmark_return = realized.shift(1).fillna(0.0)
        nav = (1.0 + strategy_return).cumprod()
        benchmark_nav = (1.0 + benchmark_return).cumprod()
        trend = data["close"].astype(float) / data["close"].astype(float).rolling(5, min_periods=1).mean() - 1.0
        drawdown = nav / nav.cummax() - 1.0
        rows.append(
            pd.DataFrame(
                {
                    "date": data["date"],
                    "model": model,
                    "nav": nav,
                    "benchmark_nav": benchmark_nav,
                    "position": position,
                    "turnover": turnover,
                    "vix": data["vix"].astype(float),
                    "trend": trend,
                    "drawdown": drawdown,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
