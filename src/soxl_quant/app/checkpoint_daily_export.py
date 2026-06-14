from __future__ import annotations

"""Helpers for persisting fold-local checkpoint replay daily paths."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    metrics = metrics_from_official_backtest(
        split_metrics=split_metrics or {},
        backtest_result=backtest_result,
    )
    row = {
        "fold": str(fold),
        "candidate": str(candidate),
        "candidate_kind": str(candidate_kind),
        "daily_path": _relative_or_absolute(path, root),
        "daily_path_rows": int(len(daily)),
        "daily_path_start": str(daily["datetime"].iloc[0]) if not daily.empty else "",
        "daily_path_end": str(daily["datetime"].iloc[-1]) if not daily.empty else "",
        "test_return": float(metrics.get("return", float("nan"))),
        "test_sharpe": float(metrics.get("sharpe", float("nan"))),
        "test_max_dd": float(metrics.get("max_dd", float("nan"))),
        "test_buy_and_hold_return": float(metrics.get("buy_and_hold_return", float("nan"))),
        "test_relative_return": float(metrics.get("relative_return", float("nan"))),
    }
    return CheckpointDailyExport(path=path, daily=daily, row=row)


__all__ = [
    "CheckpointDailyExport",
    "build_checkpoint_candidate_daily_export",
    "metrics_from_official_backtest",
]
