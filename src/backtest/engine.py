from __future__ import annotations

from math import sqrt

import pandas as pd


def signal_to_position(signal, *, min_position: float = 0.0, max_position: float = 1.0) -> pd.Series:
    return pd.Series(signal, dtype="float64").clip(lower=min_position, upper=max_position)


def run_costed_backtest(
    frame: pd.DataFrame,
    *,
    price_col: str,
    signal_col: str,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    result = frame.copy()
    result["position"] = signal_to_position(result[signal_col])
    result["asset_return"] = result[price_col].pct_change().fillna(0.0)
    result["strategy_return_before_cost"] = result["position"].shift(1).fillna(0.0) * result["asset_return"]
    result["turnover"] = result["position"].diff().abs().fillna(result["position"].abs())
    result["cost"] = result["turnover"] * ((cost_bps + slippage_bps) / 10000.0)
    result["strategy_return"] = result["strategy_return_before_cost"] - result["cost"]
    result["nav"] = (1.0 + result["strategy_return"]).cumprod()
    return result


def compare_to_benchmark(result: pd.DataFrame, *, benchmark_nav) -> dict[str, float]:
    strategy_return_pct = (float(result["nav"].iloc[-1]) / float(result["nav"].iloc[0]) - 1.0) * 100.0
    benchmark = pd.Series(benchmark_nav, dtype="float64")
    benchmark_return_pct = (float(benchmark.iloc[-1]) / float(benchmark.iloc[0]) - 1.0) * 100.0
    return {
        "return_pct": round(strategy_return_pct, 10),
        "benchmark_return_pct": round(benchmark_return_pct, 10),
        "excess_return_pct": round(strategy_return_pct - benchmark_return_pct, 10),
    }


def _max_drawdown_pct(nav: pd.Series) -> float:
    drawdown = nav / nav.cummax() - 1.0
    return round(float(drawdown.min() * 100.0), 10)


def _window_metric_row(
    frame: pd.DataFrame,
    *,
    window: str,
    start: str,
    end: str,
    benchmark_nav_col: str | None,
) -> dict[str, float | str]:
    nav = frame["nav"].astype(float)
    returns = nav.pct_change().dropna()
    return_pct = (float(nav.iloc[-1]) / float(nav.iloc[0]) - 1.0) * 100.0
    sharpe = 0.0
    if not returns.empty and float(returns.std(ddof=0)) > 0:
        sharpe = float((returns.mean() / returns.std(ddof=0)) * sqrt(252.0))
    row: dict[str, float | str] = {
        "window": window,
        "start": start,
        "end": end,
        "return_pct": round(return_pct, 10),
        "max_drawdown_pct": _max_drawdown_pct(nav),
        "sharpe": round(sharpe, 10),
        "turnover": round(float(frame.get("turnover", pd.Series(dtype="float64")).sum()), 10),
        "exposure": round(float(frame.get("position", pd.Series(dtype="float64")).abs().mean()), 10),
    }
    if benchmark_nav_col:
        benchmark = frame[benchmark_nav_col].astype(float)
        benchmark_return_pct = (float(benchmark.iloc[-1]) / float(benchmark.iloc[0]) - 1.0) * 100.0
        row["benchmark_return_pct"] = round(benchmark_return_pct, 10)
        row["excess_return_pct"] = round(return_pct - benchmark_return_pct, 10)
    return row


def stress_window_metrics(
    result: pd.DataFrame,
    *,
    windows: dict[str, tuple[str, str]],
    benchmark_nav_col: str | None = None,
) -> pd.DataFrame:
    frame = result.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    rows: list[dict[str, float | str]] = []
    for name, (start, end) in windows.items():
        window = frame.loc[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))]
        if window.empty:
            continue
        rows.append(
            _window_metric_row(
                window,
                window=name,
                start=pd.Timestamp(start).date().isoformat(),
                end=pd.Timestamp(end).date().isoformat(),
                benchmark_nav_col=benchmark_nav_col,
            )
        )
    return pd.DataFrame(rows)


def walk_forward_slices(dates, *, train_size: int, test_size: int) -> list[dict[str, str]]:
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    folds: list[dict[str, str]] = []
    start = 0
    while start + train_size + test_size <= len(index):
        train = index[start : start + train_size]
        test = index[start + train_size : start + train_size + test_size]
        folds.append(
            {
                "train_start": train[0].date().isoformat(),
                "train_end": train[-1].date().isoformat(),
                "test_start": test[0].date().isoformat(),
                "test_end": test[-1].date().isoformat(),
            }
        )
        start += test_size
    return folds


def walk_forward_metrics(
    result: pd.DataFrame,
    *,
    folds: list[dict[str, str]],
    benchmark_nav_col: str | None = None,
) -> pd.DataFrame:
    windows = {
        f"fold_{index}": (fold["test_start"], fold["test_end"])
        for index, fold in enumerate(folds, start=1)
    }
    return stress_window_metrics(result, windows=windows, benchmark_nav_col=benchmark_nav_col)
