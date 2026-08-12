from __future__ import annotations

from typing import Any

import pandas as pd
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase


def _instrument_from_index(index_value: Any, *, index_names: list[str | None], default: str | None) -> str | None:
    if isinstance(index_value, tuple):
        if "instrument" in index_names:
            return str(index_value[index_names.index("instrument")])
        return str(index_value[-1])
    if default is not None and not isinstance(index_value, str):
        return default
    return str(index_value) if index_value is not None else default


def _score_by_instrument(score: pd.Series | pd.DataFrame, *, default_instrument: str | None) -> dict[str, float]:
    if isinstance(score, pd.DataFrame):
        series = score["score"] if "score" in score.columns else score.iloc[:, 0]
    elif isinstance(score, pd.Series):
        series = score
    else:
        raise TypeError(f"score must be a pandas Series or DataFrame, got {type(score)!r}")

    values = pd.to_numeric(series, errors="coerce").dropna()
    names = list(values.index.names)
    by_instrument: dict[str, float] = {}
    for index_value, value in values.items():
        instrument = _instrument_from_index(index_value, index_names=names, default=default_instrument)
        if instrument is not None:
            by_instrument[instrument] = float(value)
    return by_instrument


def calibrated_signal_threshold(model, *, dataset, segment: str = "train", quantile: float = 0.5) -> float:
    prediction = model.predict(dataset, segment=segment)
    if isinstance(prediction, pd.DataFrame):
        scores = prediction["score"] if "score" in prediction.columns else prediction.iloc[:, 0]
    else:
        scores = prediction
    values = pd.to_numeric(scores, errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.quantile(float(quantile)))


class SignalWeightStrategy(WeightStrategyBase):
    def __init__(
        self,
        *,
        instrument: str | None = None,
        score_threshold: float = 0.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if min_weight < 0.0 or max_weight < 0.0:
            raise ValueError("target weights must be non-negative")
        if min_weight > max_weight:
            raise ValueError("min_weight must be <= max_weight")
        self.instrument = instrument
        self.score_threshold = float(score_threshold)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        scores = _score_by_instrument(score, default_instrument=self.instrument)
        instruments = [self.instrument] if self.instrument else sorted(scores)
        target: dict[str, float] = {}
        for instrument in instruments:
            if instrument is None:
                continue
            value = scores.get(instrument)
            target[instrument] = (
                self.max_weight if value is not None and value > self.score_threshold else self.min_weight
            )
        return target


class DirectTargetWeightStrategy(WeightStrategyBase):
    """Interpret each signal score as a non-negative portfolio target weight.

    This is intentionally thinner than ``SignalWeightStrategy``: qlib owns the
    signal calendar, execution loop, exchange and report generation, while the
    research workflow owns the causal target-weight construction.  Scores are
    clipped to ``[0, max_weight]`` and scaled down only when their sum exceeds
    ``risk_degree``; unallocated capital remains cash.
    """

    def __init__(self, *, max_weight: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        if max_weight <= 0.0:
            raise ValueError("max_weight must be positive")
        self.max_weight = float(max_weight)

    def generate_target_weight_position(
        self, score, current, trade_start_time, trade_end_time
    ):
        scores = _score_by_instrument(score, default_instrument=None)
        target = {
            instrument: min(self.max_weight, max(0.0, float(value)))
            for instrument, value in scores.items()
        }
        total = float(sum(target.values()))
        risk_degree = float(getattr(self, "risk_degree", 1.0))
        if total > risk_degree and total > 0.0:
            scale = risk_degree / total
            target = {instrument: value * scale for instrument, value in target.items()}
        return target
