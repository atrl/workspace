from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def quantbench_objective_families() -> list[dict[str, str]]:
    """Map QuantBench objective families to SOXL-specific targets."""
    return [
        {
            "key": "clf",
            "paper_basis": "Classification loss predicts future up/down movement.",
            "soxl_target": "fwd_positive_h",
            "metric": "clf_accuracy",
        },
        {
            "key": "ic",
            "paper_basis": "IC objective aligns predictions with future return correlation.",
            "soxl_target": "fwd_return_h",
            "metric": "ic",
        },
        {
            "key": "mse",
            "paper_basis": "MSE loss regresses future returns directly.",
            "soxl_target": "fwd_return_h",
            "metric": "mse",
        },
        {
            "key": "ranking",
            "paper_basis": "Pairwise or listwise ranking objectives prefer better-return observations.",
            "soxl_target": "relative ordering of fwd_return_h",
            "metric": "ranking_spread",
        },
        {
            "key": "utility",
            "paper_basis": "Utility maximization should reflect trading outcomes, not only prediction error.",
            "soxl_target": "position-weighted fwd_return_h with chip retention and bull capture labels",
            "metric": "utility_return",
        },
        {
            "key": "rl",
            "paper_basis": "Non-differentiable simulation and execution feedback can be optimized with RL rewards.",
            "soxl_target": "PPO reward and execution gate metrics",
            "metric": "portfolio_return_sharpe_maxdd",
        },
    ]


def _future_drawdown(close: pd.Series, horizon: int) -> pd.Series:
    values = []
    arr = close.astype(float).to_numpy()
    for idx, current in enumerate(arr):
        end = min(idx + horizon + 1, len(arr))
        if idx + horizon >= len(arr) or not math.isfinite(current) or current == 0:
            values.append(float("nan"))
            continue
        future_path = arr[idx + 1 : end]
        if len(future_path) == 0:
            values.append(float("nan"))
        else:
            values.append(float(np.nanmin(future_path) / current - 1.0))
    return pd.Series(values, index=close.index)


def build_soxl_supervised_targets(
    close: pd.Series,
    horizons: Iterable[int] = (5, 10, 20),
    chip_drawdown_floor: float = -0.20,
    bull_capture_return: float = 0.10,
) -> pd.DataFrame:
    """Build future-only labels for paper-style supervised objective screens.

    The target frame is not a tradable feature frame. It intentionally uses
    future returns/drawdowns and must only be joined to already-lagged features
    for train/validation/test objective evaluation.
    """
    close = close.astype(float).sort_index()
    out = pd.DataFrame(index=close.index)
    for horizon in tuple(int(h) for h in horizons):
        if horizon <= 0:
            raise ValueError("horizons must be positive")
        fwd_return = close.shift(-horizon) / close - 1.0
        fwd_drawdown = _future_drawdown(close, horizon)
        out[f"fwd_return_{horizon}"] = fwd_return
        out[f"fwd_positive_{horizon}"] = (fwd_return > 0.0).astype(float).where(fwd_return.notna())
        out[f"fwd_drawdown_{horizon}"] = fwd_drawdown
        out[f"chip_retention_label_{horizon}"] = (
            (fwd_drawdown >= float(chip_drawdown_floor)) & fwd_drawdown.notna()
        ).astype(float).where(fwd_drawdown.notna())
        out[f"bull_capture_label_{horizon}"] = (
            (fwd_return >= float(bull_capture_return)) & fwd_drawdown.notna()
        ).astype(float).where(fwd_return.notna())
    return out


def evaluate_prediction_objectives(
    predictions: pd.Series,
    targets: pd.DataFrame,
    horizon: int,
) -> dict[str, float]:
    """Evaluate a prediction vector against QuantBench-style SOXL objectives."""
    horizon = int(horizon)
    frame = pd.concat(
        [
            predictions.rename("prediction").astype(float),
            targets[f"fwd_return_{horizon}"].rename("fwd_return").astype(float),
            targets[f"fwd_positive_{horizon}"].rename("fwd_positive").astype(float),
            targets[f"chip_retention_label_{horizon}"].rename("chip_retention").astype(float),
            targets[f"bull_capture_label_{horizon}"].rename("bull_capture").astype(float),
        ],
        axis=1,
    ).dropna()
    if frame.empty:
        return {
            "rows": 0.0,
            "clf_accuracy": float("nan"),
            "ic": float("nan"),
            "mse": float("nan"),
            "ranking_spread": float("nan"),
            "utility_return": float("nan"),
        }

    pred = frame["prediction"]
    ret = frame["fwd_return"]
    positive = frame["fwd_positive"]
    pred_positive = (pred > 0.0).astype(float)
    clf_accuracy = float((pred_positive == positive).mean())
    ic = float(pred.corr(ret)) if pred.std(ddof=0) > 0 and ret.std(ddof=0) > 0 else float("nan")
    mse = float(((pred - ret) ** 2).mean())

    high = frame[pred >= pred.quantile(0.80)]["fwd_return"].mean()
    low = frame[pred <= pred.quantile(0.20)]["fwd_return"].mean()
    ranking_spread = float(high - low)

    clipped_position = pred.clip(lower=0.0, upper=1.0)
    retention_adjustment = 0.50 + 0.50 * frame["chip_retention"]
    bull_adjustment = 1.0 + 0.25 * frame["bull_capture"]
    utility_return = float((clipped_position * retention_adjustment * bull_adjustment * ret).mean())

    return {
        "rows": float(len(frame)),
        "clf_accuracy": round(clf_accuracy, 6),
        "ic": ic,
        "mse": mse,
        "ranking_spread": ranking_spread,
        "utility_return": utility_return,
    }


__all__ = [
    "build_soxl_supervised_targets",
    "evaluate_prediction_objectives",
    "quantbench_objective_families",
]
