from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.model import ModelSpec, run_qlib_model_selection


LABEL = "Ref($close, -1) / $close - 1"


class SegmentDataset:
    def __init__(self) -> None:
        index = pd.Index(range(3), name="row")
        self.frames = {
        "train": pd.DataFrame({"$close": [1.0, 2.0, 3.0], LABEL: [0.0, 0.0, 0.0]}, index=index),
            "valid": pd.DataFrame({"$close": [1.0, 2.0, 3.0], LABEL: [1.0, 2.0, 3.0]}, index=index),
            "test": pd.DataFrame({"$close": [10.0, 20.0, 30.0], LABEL: [10.0, 20.0, 30.0]}, index=index),
        }

    def prepare(self, segment: str) -> pd.DataFrame:
        return self.frames[segment].copy()


class SegmentPredictionModel:
    qlib_dataset_api = ["DatasetH", "DataHandlerLP"]

    def __init__(self, *, name: str, predictions: dict[str, list[float]]) -> None:
        self.name = name
        self.predictions = predictions
        self.implementation = f"fake_{name}"
        self.fit_segments: list[str] = []
        self.predict_segments: list[str] = []

    def fit(self, dataset, *, segment: str = "train") -> "SegmentPredictionModel":
        self.fit_segments.append(segment)
        return self

    def predict(self, dataset, *, segment: str = "test") -> pd.Series:
        self.predict_segments.append(segment)
        frame = dataset.prepare(segment)
        return pd.Series(self.predictions[segment], index=frame.index, name="score")

    def save(self, path: Path, *, model_name: str) -> None:
        path.write_text(
            json.dumps({"model": model_name, "implementation": self.implementation}, sort_keys=True),
            encoding="utf-8",
        )


def test_model_selection_chooses_validation_winner_not_test_winner(monkeypatch, tmp_path):
    models = {
        "valid_winner": SegmentPredictionModel(
            name="valid_winner",
            predictions={
                "train": [0.0, 0.5, 1.0],
                "valid": [1.0, 2.0, 3.0],
                "test": [0.0, 0.0, 0.0],
            },
        ),
        "test_winner": SegmentPredictionModel(
            name="test_winner",
            predictions={
                "train": [0.0, 0.5, 1.0],
                "valid": [0.0, 0.0, 0.0],
                "test": [10.0, 20.0, 30.0],
            },
        ),
    }

    def fake_builder(spec: ModelSpec) -> SegmentPredictionModel:
        return models[spec.name]

    monkeypatch.setattr("src.model.registry.build_model_from_spec", fake_builder)

    result = run_qlib_model_selection(
        SegmentDataset(),
        specs=[
            ModelSpec("valid_winner", "fake", "mse", {"uses_qlib_dataset": True}),
            ModelSpec("test_winner", "fake", "mse", {"uses_qlib_dataset": True}),
        ],
        output_dir=tmp_path,
        selection_metric="objective_score",
        selection_segment="valid",
        report_segment="test",
    )

    rows = result.model_metrics.set_index("model")

    assert result.selected_spec.name == "valid_winner"
    assert bool(rows.loc["valid_winner", "selected"]) is True
    assert rows.loc["valid_winner", "selection_objective_score"] > rows.loc["test_winner", "selection_objective_score"]
    assert rows.loc["valid_winner", "test_objective_score"] < rows.loc["test_winner", "test_objective_score"]
    assert rows["selection_segment"].eq("valid").all()
    assert rows["report_segment"].eq("test").all()
    assert rows["signal_threshold_source"].eq("train_prediction_median").all()
    assert rows["train_signal_threshold"].tolist() == [0.5, 0.5]
    assert models["valid_winner"].fit_segments == ["train"]
    assert models["test_winner"].fit_segments == ["train"]
    assert models["valid_winner"].predict_segments == ["train", "valid", "test"]
    assert models["test_winner"].predict_segments == ["train", "valid", "test"]
    assert (tmp_path / "model_selection.csv").exists()
    assert (tmp_path / "all_model_predictions.csv").exists()
