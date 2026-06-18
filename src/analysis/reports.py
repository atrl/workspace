from __future__ import annotations

import math
from typing import Mapping

import pandas as pd

from src.objective import utility_score


def _max_drawdown_pct(nav: pd.Series) -> float:
    values = nav.astype(float)
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    return abs(float(drawdown.min())) * 100.0


def _return_pct(nav: pd.Series) -> float:
    values = nav.astype(float)
    if values.empty or abs(float(values.iloc[0])) < 1e-12:
        return 0.0
    return (float(values.iloc[-1]) / float(values.iloc[0]) - 1.0) * 100.0


def _sharpe(nav: pd.Series) -> float:
    returns = nav.astype(float).pct_change().dropna()
    if returns.empty:
        return 0.0
    std = float(returns.std(ddof=0))
    if std <= 1e-12 or not math.isfinite(std):
        return 0.0
    return float(returns.mean()) / std * math.sqrt(252.0)


def _classify_regime(row: pd.Series) -> str:
    vix = float(row.get("vix", 0.0))
    trend = float(row.get("trend", 0.0))
    drawdown = float(row.get("drawdown", 0.0))
    if vix >= 30.0 or drawdown <= -0.10:
        return "stress"
    if trend < 0.0 or drawdown <= -0.03:
        return "downtrend"
    if vix < 20.0 and trend > 0.0:
        return "bull"
    return "neutral"


def _metrics(group: pd.DataFrame) -> dict[str, float]:
    nav = group["nav"].astype(float)
    benchmark_nav = group.get("benchmark_nav")
    return_pct = _return_pct(nav)
    benchmark_return = _return_pct(benchmark_nav.astype(float)) if benchmark_nav is not None else 0.0
    max_drawdown_pct = _max_drawdown_pct(nav)
    sharpe = _sharpe(nav)
    turnover = float(group.get("turnover", pd.Series([0.0])).astype(float).sum())
    score = utility_score(
        return_pct=return_pct,
        sharpe=sharpe,
        max_drawdown_pct=max_drawdown_pct,
        turnover=turnover,
    )
    return {
        "return_pct": return_pct,
        "benchmark_return_pct": benchmark_return,
        "relative_return_pct": return_pct - benchmark_return,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "avg_position": float(group.get("position", pd.Series([0.0])).astype(float).mean()),
        "turnover": turnover,
        "stability_score": score,
    }


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "model", "nav"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"analysis frame missing columns: {missing}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    if "regime" not in result.columns:
        result["regime"] = result.apply(_classify_regime, axis=1)
    return result.sort_values(["model", "date"]).reset_index(drop=True)


def compute_regime_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare(frame)
    rows: list[dict[str, object]] = []
    for (model, regime), group in prepared.groupby(["model", "regime"], sort=True):
        rows.append({"model": model, "regime": regime, **_metrics(group)})
    return pd.DataFrame(rows).sort_values(["model", "regime"]).reset_index(drop=True)


def compute_window_metrics(frame: pd.DataFrame, *, windows: Mapping[str, tuple[str, str]]) -> pd.DataFrame:
    prepared = _prepare(frame)
    rows: list[dict[str, object]] = []
    for window, (start, end) in windows.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        window_frame = prepared.loc[(prepared["date"] >= start_ts) & (prepared["date"] <= end_ts)]
        for model, group in window_frame.groupby("model", sort=True):
            rows.append({"window": window, "model": model, **_metrics(group)})
    return pd.DataFrame(rows).sort_values(["window", "model"]).reset_index(drop=True)


def compute_window_regime_metrics(frame: pd.DataFrame, *, windows: Mapping[str, tuple[str, str]]) -> pd.DataFrame:
    prepared = _prepare(frame)
    rows: list[dict[str, object]] = []
    for window, (start, end) in windows.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        window_frame = prepared.loc[(prepared["date"] >= start_ts) & (prepared["date"] <= end_ts)]
        for (model, regime), group in window_frame.groupby(["model", "regime"], sort=True):
            rows.append(
                {
                    "window": window,
                    "regime": regime,
                    "segment": f"{window}::{regime}",
                    "model": model,
                    **_metrics(group),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "window",
                "regime",
                "segment",
                "model",
                "return_pct",
                "benchmark_return_pct",
                "relative_return_pct",
                "max_drawdown_pct",
                "sharpe",
                "avg_position",
                "turnover",
                "stability_score",
            ]
        )
    return pd.DataFrame(rows).sort_values(["window", "regime", "model"]).reset_index(drop=True)


def compare_models(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare(frame)
    rows = [{"model": model, **_metrics(group)} for model, group in prepared.groupby("model", sort=True)]
    result = pd.DataFrame(rows).sort_values(
        ["stability_score", "relative_return_pct", "return_pct"],
        ascending=[False, False, False],
    )
    result.insert(0, "rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)
