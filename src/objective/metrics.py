from __future__ import annotations

import pandas as pd


def _series(values) -> pd.Series:
    return pd.Series(values, dtype="float64")


def classification_accuracy(labels, predictions) -> float:
    actual = list(labels)
    pred = list(predictions)
    if len(actual) != len(pred):
        raise ValueError("labels and predictions must have the same length")
    return sum(1 for left, right in zip(actual, pred, strict=True) if left == right) / len(actual)


def mean_squared_error(labels, predictions) -> float:
    diff = _series(labels) - _series(predictions)
    return float((diff**2).mean())


def information_coefficient(signal, forward_return) -> float:
    return round(float(_series(signal).corr(_series(forward_return), method="pearson")), 10)


def rank_ic(signal, forward_return) -> float:
    return round(float(_series(signal).corr(_series(forward_return), method="spearman")), 10)


def utility_score(*, return_pct: float, max_drawdown_pct: float, turnover: float) -> float:
    return round(float(return_pct - abs(max_drawdown_pct) * 0.5 - turnover * 2.0), 10)
