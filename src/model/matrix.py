from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.analysis import compare_models, compute_regime_metrics, compute_window_metrics
from src.backtest import backtest_prediction_frame
from src.data import build_supervised_dataset, load_soxl_support_frame
from src.model.registry import QUANTBENCH_MODEL_NAMES
from src.model.training import fit_predict_model


@dataclass(frozen=True)
class QuantBenchMatrixOutputs:
    predictions: pd.DataFrame
    daily: pd.DataFrame
    regime_metrics: pd.DataFrame
    window_metrics: pd.DataFrame
    model_comparison: pd.DataFrame
    summary: dict


def run_quantbench_matrix_smoke(
    *,
    card_dir: Path,
    start: str,
    end: str,
    train_end: str,
    model_names: Iterable[str] = QUANTBENCH_MODEL_NAMES,
) -> QuantBenchMatrixOutputs:
    card = Path(card_dir)
    card.mkdir(parents=True, exist_ok=True)
    price_frame = load_soxl_support_frame(start=start, end=end)
    dataset = build_supervised_dataset(price_frame, horizon=1)
    train_end_ts = pd.Timestamp(train_end)
    train = dataset.frame.loc[dataset.frame["date"] <= train_end_ts].copy()
    test = dataset.frame.loc[dataset.frame["date"] > train_end_ts].copy()
    if train.empty or test.empty:
        raise ValueError("train/test split produced an empty side")

    prediction_frames = [
        fit_predict_model(name, train=train, test=test, feature_columns=dataset.feature_columns)
        for name in tuple(model_names)
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    daily = backtest_prediction_frame(predictions, price_frame=price_frame)
    regime_metrics = compute_regime_metrics(daily)
    window_metrics = compute_window_metrics(daily, windows={"test": (str(test["date"].min().date()), str(test["date"].max().date()))})
    model_comparison = compare_models(daily)
    summary = {
        "status": "smoke_completed",
        "model_count": int(predictions["model"].nunique()),
        "start": start,
        "end": end,
        "train_end": train_end,
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "top_model": str(model_comparison.iloc[0]["model"]),
        "adapter_status": "smoke_adapter",
        "note": "Smoke adapters validate the full matrix contract; paper-strict replacements can be swapped per ModelSpec.",
    }

    predictions.to_csv(card / "predictions.csv", index=False)
    daily.to_csv(card / "daily_backtest.csv", index=False)
    regime_metrics.to_csv(card / "regime_metrics.csv", index=False)
    window_metrics.to_csv(card / "window_metrics.csv", index=False)
    model_comparison.to_csv(card / "model_comparison.csv", index=False)
    (card / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return QuantBenchMatrixOutputs(
        predictions=predictions,
        daily=daily,
        regime_metrics=regime_metrics,
        window_metrics=window_metrics,
        model_comparison=model_comparison,
        summary=summary,
    )
