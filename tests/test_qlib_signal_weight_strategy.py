from __future__ import annotations

import pandas as pd


def test_signal_weight_strategy_maps_single_asset_scores_to_target_weights():
    from qlib.contrib.strategy.signal_strategy import WeightStrategyBase

    from src.backtest.strategy import SignalWeightStrategy

    assert issubclass(SignalWeightStrategy, WeightStrategyBase)

    strategy = SignalWeightStrategy(
        signal=pd.Series(dtype="float64"),
        instrument="SOXL",
        score_threshold=0.0,
        min_weight=0.0,
        max_weight=1.0,
    )

    assert strategy.generate_target_weight_position(
        score=pd.Series([0.01], index=pd.Index(["SOXL"], name="instrument")),
        current=None,
        trade_start_time=pd.Timestamp("2026-01-06"),
        trade_end_time=pd.Timestamp("2026-01-06"),
    ) == {"SOXL": 1.0}
    assert strategy.generate_target_weight_position(
        score=pd.DataFrame(
            {"score": [-0.01]},
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2026-01-05"), "SOXL")],
                names=["datetime", "instrument"],
            ),
        ),
        current=None,
        trade_start_time=pd.Timestamp("2026-01-06"),
        trade_end_time=pd.Timestamp("2026-01-06"),
    ) == {"SOXL": 0.0}
    assert strategy.generate_target_weight_position(
        score=pd.DataFrame({"not_score": [0.0]}, index=pd.Index(["SOXL"], name="instrument")),
        current=None,
        trade_start_time=pd.Timestamp("2026-01-06"),
        trade_end_time=pd.Timestamp("2026-01-06"),
    ) == {"SOXL": 0.0}


def test_score_threshold_is_calibrated_from_train_predictions_without_test_leakage():
    from src.backtest.strategy import calibrated_signal_threshold

    class RecordingModel:
        def __init__(self):
            self.segments: list[str] = []

        def predict(self, dataset, *, segment: str = "test"):
            self.segments.append(segment)
            values = {
                "train": [-0.02, 0.01, 0.03],
                "test": [1.0, 2.0, 3.0],
            }[segment]
            return pd.Series(values, name="score")

    model = RecordingModel()

    assert calibrated_signal_threshold(model, dataset=object(), segment="train") == 0.01
    assert model.segments == ["train"]
