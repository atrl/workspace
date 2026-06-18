from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.objective import utility_score


@dataclass(frozen=True)
class CostModel:
    open_cost_bps: float = 0.0
    close_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    schema_version: str = "backtest.cost_model.v1"

    def transaction_cost_return(self, position_delta: pd.Series) -> pd.Series:
        delta = position_delta.astype(float).fillna(0.0)
        buy_turnover = delta.clip(lower=0.0)
        sell_turnover = (-delta).clip(lower=0.0)
        buy_cost = (float(self.open_cost_bps) + float(self.slippage_bps)) / 10000.0
        sell_cost = (float(self.close_cost_bps) + float(self.slippage_bps)) / 10000.0
        return -(buy_turnover * buy_cost + sell_turnover * sell_cost)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PositionPolicy:
    min_position: float = 0.0
    max_position: float = 1.0
    neutral_position: float = 0.5
    signal_scale: float | None = None
    lag: int = 1
    schema_version: str = "backtest.position_policy.v1"

    def signal_to_position(self, prediction: pd.Series) -> pd.Series:
        signal = prediction.astype(float).fillna(0.0)
        scale = float(self.signal_scale) if self.signal_scale is not None else _robust_signal_scale(signal)
        raw = float(self.neutral_position) + signal / (4.0 * scale)
        return raw.clip(lower=float(self.min_position), upper=float(self.max_position))

    def execution_position(self, prediction: pd.Series) -> pd.Series:
        signal_position = self.signal_to_position(prediction)
        if int(self.lag) <= 0:
            return signal_position
        return signal_position.shift(int(self.lag)).fillna(float(self.neutral_position))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestConfig:
    start: str
    end: str
    benchmark: str = "SOXL"
    position_policy: PositionPolicy = PositionPolicy()
    cost_model: CostModel = CostModel()
    stress_windows: Mapping[str, tuple[str, str]] | None = None
    schema_version: str = "backtest.config.v1"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stress_windows"] = {
            str(name): [str(start), str(end)] for name, (start, end) in dict(self.stress_windows or {}).items()
        }
        return payload


@dataclass(frozen=True)
class WalkForwardSplit:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    schema_version: str = "backtest.walk_forward_split.v1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    model_summary: pd.DataFrame
    config: BacktestConfig
    schema_version: str = "backtest.result.v1"

    def to_analysis_frame(self) -> pd.DataFrame:
        return self.daily.copy()

    def summary_payload(self) -> dict[str, object]:
        top_model = ""
        if not self.model_summary.empty and "model" in self.model_summary.columns:
            top_model = str(self.model_summary.iloc[0]["model"])
        return {
            "schema_version": self.schema_version,
            "status": "backtest_completed",
            "config": self.config.to_dict(),
            "model_count": int(self.daily["model"].nunique()) if "model" in self.daily.columns else 0,
            "row_count": int(len(self.daily)),
            "top_model": top_model,
        }


def _robust_signal_scale(signal: pd.Series) -> float:
    finite = signal.replace([np.inf, -np.inf], np.nan).dropna().abs()
    if finite.empty:
        return 1.0
    scale = float(finite.median())
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(finite.mean())
    if not np.isfinite(scale) or scale <= 1e-12:
        return 1.0
    return scale


def _stress_label(date: pd.Timestamp, stress_windows: Mapping[str, tuple[str, str]] | None) -> str | None:
    for name, (start, end) in dict(stress_windows or {}).items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return str(name)
    return None


def _max_drawdown_pct(nav: pd.Series) -> float:
    values = nav.astype(float)
    drawdown = values / values.cummax() - 1.0
    return abs(float(drawdown.min())) * 100.0


def _sharpe_from_returns(returns: pd.Series) -> float:
    values = returns.astype(float).dropna()
    if values.empty:
        return 0.0
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return 0.0
    return float(values.mean()) / std * np.sqrt(252.0)


def _model_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in daily.groupby("model", sort=True):
        ordered = group.sort_values("date")
        nav = ordered["nav"].astype(float)
        benchmark_nav = ordered["benchmark_nav"].astype(float)
        return_pct = (float(nav.iloc[-1]) / float(nav.iloc[0]) - 1.0) * 100.0
        benchmark_return_pct = (float(benchmark_nav.iloc[-1]) / float(benchmark_nav.iloc[0]) - 1.0) * 100.0
        max_drawdown_pct = _max_drawdown_pct(nav)
        sharpe = _sharpe_from_returns(ordered["net_strategy_return"])
        avg_turnover = float(ordered["turnover"].astype(float).mean())
        turnover = float(ordered["turnover"].astype(float).sum())
        score = utility_score(
            return_pct=return_pct,
            sharpe=sharpe,
            max_drawdown_pct=max_drawdown_pct,
            turnover=turnover,
        )
        rows.append(
            {
                "model": str(model),
                "return_pct": return_pct,
                "benchmark_return_pct": benchmark_return_pct,
                "relative_return_pct": return_pct - benchmark_return_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "sharpe": sharpe,
                "avg_position": float(ordered["position"].astype(float).mean()),
                "avg_turnover": avg_turnover,
                "turnover": turnover,
                "utility_score": score,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["utility_score", "relative_return_pct", "return_pct"],
        ascending=[False, False, False],
    )
    result.insert(0, "rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def _prepare_inputs(
    predictions: pd.DataFrame,
    price_frame: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_predictions = {"date", "model", "prediction", "target_return"}
    missing = sorted(required_predictions - set(predictions.columns))
    if missing:
        raise ValueError(f"prediction frame missing columns: {missing}")
    required_prices = {"date", "close", "vix"}
    missing_prices = sorted(required_prices - set(price_frame.columns))
    if missing_prices:
        raise ValueError(f"price frame missing columns: {missing_prices}")

    start = pd.Timestamp(config.start)
    end = pd.Timestamp(config.end)
    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pred = pred.loc[(pred["date"] >= start) & (pred["date"] <= end)].copy()
    prices = price_frame.loc[:, ["date", "close", "vix"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.loc[(prices["date"] >= start) & (prices["date"] <= end)].copy()
    return pred, prices


def run_prediction_backtest(
    predictions: pd.DataFrame,
    *,
    price_frame: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    pred, prices = _prepare_inputs(predictions, price_frame, config)
    rows: list[pd.DataFrame] = []
    for model, group in pred.groupby("model", sort=True):
        data = group.merge(prices, on="date", how="left").sort_values("date").reset_index(drop=True)
        position = config.position_policy.execution_position(data["prediction"])
        position_delta = position.diff().fillna(position - float(config.position_policy.neutral_position))
        turnover = position_delta.abs()
        cost_return = config.cost_model.transaction_cost_return(position_delta)
        realized = data["target_return"].astype(float).fillna(0.0)
        gross_return = position * realized
        net_return = gross_return + cost_return
        benchmark_return = realized
        nav = (1.0 + net_return).cumprod()
        benchmark_nav = (1.0 + benchmark_return).cumprod()
        close = data["close"].astype(float)
        trend = close / close.rolling(5, min_periods=1).mean() - 1.0
        drawdown = nav / nav.cummax() - 1.0
        model_frame = pd.DataFrame(
            {
                "date": data["date"],
                "model": str(model),
                "prediction": data["prediction"].astype(float),
                "target_return": realized,
                "gross_strategy_return": gross_return,
                "cost_return": cost_return,
                "net_strategy_return": net_return,
                "benchmark_return": benchmark_return,
                "nav": nav,
                "benchmark_nav": benchmark_nav,
                "position": position,
                "turnover": turnover,
                "vix": data["vix"].astype(float),
                "trend": trend,
                "drawdown": drawdown,
            }
        )
        model_frame["stress_window"] = model_frame["date"].map(
            lambda value: _stress_label(pd.Timestamp(value), config.stress_windows)
        )
        rows.append(model_frame)
    if rows:
        daily = pd.concat(rows, ignore_index=True).sort_values(["model", "date"]).reset_index(drop=True)
    else:
        daily = pd.DataFrame()
    return BacktestResult(daily=daily, model_summary=_model_summary(daily) if not daily.empty else pd.DataFrame(), config=config)


def build_walk_forward_splits(
    dates: pd.Series | pd.DatetimeIndex | list[str],
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> list[WalkForwardSplit]:
    ordered = pd.Series(pd.to_datetime(list(dates))).drop_duplicates().sort_values().reset_index(drop=True)
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = int(step_size or test_size)
    if step <= 0:
        raise ValueError("step_size must be positive")

    splits: list[WalkForwardSplit] = []
    start_idx = 0
    fold = 1
    while start_idx + train_size + test_size <= len(ordered):
        train = ordered.iloc[start_idx : start_idx + train_size]
        test = ordered.iloc[start_idx + train_size : start_idx + train_size + test_size]
        splits.append(
            WalkForwardSplit(
                fold=fold,
                train_start=str(train.iloc[0].date()),
                train_end=str(train.iloc[-1].date()),
                test_start=str(test.iloc[0].date()),
                test_end=str(test.iloc[-1].date()),
            )
        )
        fold += 1
        start_idx += step
    return splits
