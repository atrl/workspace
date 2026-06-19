from __future__ import annotations

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
