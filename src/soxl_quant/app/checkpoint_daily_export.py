from __future__ import annotations

"""Helpers for persisting fold-local checkpoint replay daily paths."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.soxl_quant.core.metric_reconciliation import slice_warm_start_daily_path


@dataclass(frozen=True)
class CheckpointDailyExport:
    path: Path
    daily: pd.DataFrame
    row: dict[str, Any]


def metrics_from_official_backtest(
    *,
    split_metrics: dict[str, Any],
    backtest_result: dict[str, Any],
) -> dict[str, float]:
    """Return headline metrics, preferring qlib official backtest fields."""

    out = dict(split_metrics)
    for src, dst in [
        ("return", "return"),
        ("sharpe", "sharpe"),
        ("max_dd", "max_dd"),
        ("bnh_return", "buy_and_hold_return"),
        ("relative_outperformance", "relative_return"),
    ]:
        if src in backtest_result:
            out[dst] = float(backtest_result[src])
    return out


def _pct_return(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return 0.0
    start = float(clean.iloc[0])
    end = float(clean.iloc[-1])
    if not np.isfinite(start) or start == 0.0:
        return float("nan")
    return float((end / start - 1.0) * 100.0)


def _max_drawdown_pct(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    peak = clean.cummax()
    drawdown = clean / peak - 1.0
    return float(-drawdown.min() * 100.0)


def metrics_from_daily_path(daily: pd.DataFrame) -> dict[str, float]:
    """Return metrics from the exact rebased daily path written to disk."""

    if daily.empty:
        return {
            "return": 0.0,
            "max_dd": 0.0,
            "avg_pos": 0.0,
            "buy_and_hold_return": 0.0,
            "relative_return": 0.0,
        }
    strategy_return = _pct_return(daily["strategy_nav"])
    buyhold_return = _pct_return(daily["buyhold_nav"])
    return {
        "return": strategy_return,
        "max_dd": _max_drawdown_pct(daily["strategy_nav"]),
        "avg_pos": float(pd.to_numeric(daily["position"], errors="coerce").fillna(0.0).mean()),
        "buy_and_hold_return": buyhold_return,
        "relative_return": float(strategy_return - buyhold_return),
    }


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return stem.strip("._") or "candidate"


def _relative_or_absolute(path: Path, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_checkpoint_candidate_daily_export(
    *,
    backtest_result: dict[str, Any],
    price_series: pd.Series,
    output_dir: Path,
    fold: str,
    candidate: str,
    candidate_kind: str,
    split_metrics: dict[str, Any] | None = None,
    root: Path | None = None,
) -> CheckpointDailyExport:
    """Persist an official replay daily path and return candidate row fields."""

    if "portfolio" not in backtest_result:
        raise ValueError("backtest_result missing portfolio")
    eval_seg = tuple(backtest_result.get("effective_test_seg") or ())
    if len(eval_seg) != 2:
        raise ValueError("backtest_result missing effective_test_seg")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_safe_stem(fold)}__{_safe_stem(candidate)}_official_daily_path.csv"
    daily = slice_warm_start_daily_path(
        portfolio_df=backtest_result["portfolio"],
        price_series=price_series,
        eval_seg=(str(eval_seg[0]), str(eval_seg[1])),
        position_history=backtest_result.get("position_history"),
        trade_signals=backtest_result.get("trade_signals"),
    )
    daily.to_csv(path, index=False)

    raw_metrics = metrics_from_official_backtest(
        split_metrics=split_metrics or {},
        backtest_result=backtest_result,
    )
    daily_metrics = metrics_from_daily_path(daily)
    diffs = {
        key: abs(float(daily_metrics[key]) - float(raw_metrics[key]))
        for key in ("return", "max_dd", "buy_and_hold_return", "relative_return")
        if key in raw_metrics
    }
    row = {
        "fold": str(fold),
        "candidate": str(candidate),
        "candidate_kind": str(candidate_kind),
        "daily_path": _relative_or_absolute(path, root),
        "daily_path_rows": int(len(daily)),
        "daily_path_start": str(daily["datetime"].iloc[0]) if not daily.empty else "",
        "daily_path_end": str(daily["datetime"].iloc[-1]) if not daily.empty else "",
        "metric_basis": "official_daily_path_rebased_eval_window",
        "raw_metric_basis": "backtest_profile_reported_metrics",
        "test_return": float(daily_metrics.get("return", float("nan"))),
        "test_sharpe": float(raw_metrics.get("sharpe", float("nan"))),
        "test_max_dd": float(daily_metrics.get("max_dd", float("nan"))),
        "test_avg_pos": float(daily_metrics.get("avg_pos", float("nan"))),
        "test_buy_and_hold_return": float(daily_metrics.get("buy_and_hold_return", float("nan"))),
        "test_relative_return": float(daily_metrics.get("relative_return", float("nan"))),
        "raw_backtest_return": float(raw_metrics.get("return", float("nan"))),
        "raw_backtest_sharpe": float(raw_metrics.get("sharpe", float("nan"))),
        "raw_backtest_max_dd": float(raw_metrics.get("max_dd", float("nan"))),
        "raw_backtest_buy_and_hold_return": float(raw_metrics.get("buy_and_hold_return", float("nan"))),
        "raw_backtest_relative_return": float(raw_metrics.get("relative_return", float("nan"))),
        "metric_return_abs_diff": float(diffs.get("return", 0.0)),
        "metric_max_dd_abs_diff": float(diffs.get("max_dd", 0.0)),
        "metric_buyhold_abs_diff": float(diffs.get("buy_and_hold_return", 0.0)),
        "metric_relative_abs_diff": float(diffs.get("relative_return", 0.0)),
    }
    return CheckpointDailyExport(path=path, daily=daily, row=row)


__all__ = [
    "CheckpointDailyExport",
    "build_checkpoint_candidate_daily_export",
    "metrics_from_daily_path",
    "metrics_from_official_backtest",
]
